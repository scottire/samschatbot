from pprint import pprint
import dotenv
import os
import json
from openai import OpenAI
from langsmith.run_helpers import traceable
import requests
from datetime import datetime

dotenv.load_dotenv()

if os.getenv('IS_CLOUD') == 'true':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI()

import chromadb
chroma_client = chromadb.PersistentClient('./chroma.db')


def query_articles(query_text):
    q = chroma_client.get_collection("stratechery_articles").query(query_texts=query_text, n_results=7)
    return q


def get_articles_info_from_json(json_file_name):
    with open(json_file_name, 'r') as file:
        articles = json.load(file)
    articles_format = []
    article_titles = [article['title'] for article in articles]
    for i, article in enumerate(articles):
        article['publish_date'] = datetime.strptime(article['publish_date'], "%a, %d %b %Y %H:%M:%S %z").strftime("%b %d, %Y")
        articles_format.append(f"{i+1}. {article['title']} ({article['publish_date']})")
    most_recent_article_title = articles[0]['title']
    most_recent_article_date = articles[0]['publish_date']
    most_recent_article_url = articles[0]['public_url']
    oldest_article_date = articles[-1]['publish_date']
    return (len(articles), articles_format, article_titles,
            most_recent_article_title, most_recent_article_date, most_recent_article_url, oldest_article_date)


(NUM_ARTICLES, ARTICLES_FORMAT, ARTICLE_TITLES,
 MOST_RECENT_ARTICLE_TITLE, MOST_RECENT_ARTICLE_DATE, MOST_RECENT_ARTICLE_URL, OLDEST_ARTICLE_DATE) = get_articles_info_from_json('data.json')


SYSTEM_MESSAGE = f"""* You are a bot that knows everything about Ben Thompson's Stratechery articles (https://stratechery.com/). You are smart, witty, and love tech! You talk candidly and casually.
* You are trained on Stratechery articles since Nov 6, 2023. Ben Wallace (https://ben-wallace.replit.app/) created you. Your code can be found at https://github.com/benfwalla/BenThompsonChatbot. You are not approved by Ben Thompson.
* You are trained on the {NUM_ARTICLES} most recent Stratechery articles. The oldest article is {OLDEST_ARTICLE_DATE}. Here are their names and publish dates from most recent to oldest: {ARTICLES_FORMAT}
* You will answer questions using Stratechery articles. You will always refer to the specific name of the article you are citing and hyperlink to its url, as such: [Article Title](Article URL).
* If you are referring to Ben Thompson, just say "Ben". If you can't answer, you will explain why and suggest sending the question to email@sharptech.fm where Ben can answer it directly!
* A user's questions may be followed by a bunch of possible answers from Stratechery articles. Each article is is separated by `-----` and is formatted as such: `[Article Title](Article URL)\\n[Chunk of Article Content]`. Use your best judgement to answer the user's query based on the articles provided.
* If you are asked to provide text to an entire article, kindly decline and explain that Stratechery is a premium offering and you can sign up at https://stratechery.com/.
* Facts about Stratechery: Stratechery provides analysis on the business, strategy, and impact of technology and media, and how technology is changing society. It's known for its weekly articles, which are free, and three subscriber-only Daily Updates per week. The site is recommended by The New York Times and has subscribers from over 85 countries, including executives, venture capitalists, and tech enthusiasts. The Stratechery Plus bundle includes the Stratechery Update, Stratechery Interviews, Dithering podcast, Sharp Tech podcast, Sharp China podcast, and Greatest Of All Talk podcast, available for $12 per month, or $120 per year.
* Facts about Ben Thompson: Ben Thompson is the author of Stratechery. He's based in Taipei, Taiwan, and has worked at companies like Apple (interned), Microsoft, and Automattic. He holds an MBA from Kellogg School of Management and an MEM from McCormick Engineering school. Ben has been writing Stratechery since 2013, and it has been his full-time job since 2014. You can follow him on X @benthompson (https://x.com/benthompson)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_article_chunks_for_rag",
            "description": "This function provides chunks of text from Stratechery articles that are relevant to the query. "
                           "ONLY use this function if the existing information in your message history is not enough.",
            "parameters": {
                "type": "object",
                "properties": {
                  "articles": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ARTICLE_TITLES
                    },
                    "description": "A list of articles you think is most relevant to the given query from your system message. Provide no more than the top 3 most likely and recent (e.g. ['Aggregator's AI Risk', 'An Interview with Nat Friedman and Daniel Gross Reasoning About AI']).",
                  }
                },
                "required": ["articles"],
            },
        }
    }
]


@traceable(run_type="llm")
def call_openai(messages, model="gpt-3.5-turbo"):
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS
    )


def fetch_article_summaries(articles_to_summarize, json_file_name='data.json'):
    """Returns a list of dicts with article titles, summaries, and URLs"""
    with open(json_file_name, 'r') as file:
        articles = json.load(file)

    summaries = []
    for article_title in articles_to_summarize:
        article_dict = next(item for item in articles if item["title"] == article_title)
        summaries.append({
            "title": article_title,
            "summary": article_dict["summary"],
            "url": article_dict["public_url"]
        })

    return summaries


def fetch_article_chunks_from_query_search(query_text):
    """Returns an organized list of article chunks from a given query"""
    q = query_articles(query_text)

    documents = q['documents'][0]
    distances = q['distances'][0]
    metadatas = q['metadatas'][0]
    ids = q['ids'][0]

    combined = list(zip(distances, documents, metadatas, ids))
    combined.sort(key=lambda x: x[0])

    grouped_chunks = {}
    for distance, document, metadata, article_id in combined:
        article_title = metadata['title']
        if article_title not in grouped_chunks:
            grouped_chunks[article_title] = {'url': metadata['url'], 'documents': []}
        grouped_chunks[article_title]['documents'].append(document)

    return grouped_chunks


def combine_summaries_and_chunks(summaries, chunks):
    """Combines article summaries and chunks into a single string"""
    result = []
    for summary in summaries:
        title = summary['title']
        result.append(f"[{title}]({summary['url']})")
        result.append(f"Summary: {summary['summary']}")
        if title in chunks:
            result.extend(chunks[title]['documents'])
            del chunks[title]  # Remove the article from chunks to avoid duplication
        result.append("-----")

    # Append any remaining chunks that didn't have a summary
    for title, info in chunks.items():
        result.append(f"[{title}]({info['url']})")
        result.extend(info['documents'])
        result.append("-----")

    return "\n".join(result).strip("-----\n")


@traceable(run_type="chain")
def create_chat_completion_with_rag(query_text, message_chain, openai_model):
    message_chain.append({"role": "user", "content": query_text})

    completion = call_openai(message_chain)
    response_message = completion.choices[0].message
    tool_calls = response_message.tool_calls
    print("\n\n")
    print(tool_calls)
    if tool_calls and tool_calls[0].function.name == "fetch_article_chunks_for_rag":
        if 'articles' in tool_calls[0].function.arguments:
            article_titles = json.loads(tool_calls[0].function.arguments)['articles']
            article_summaries = fetch_article_summaries(article_titles)
        else:
            article_summaries = []

        article_chunks = fetch_article_chunks_from_query_search(query_text)
        combined_content = combine_summaries_and_chunks(article_summaries, article_chunks)

        message_chain.append(response_message)
        message_chain.append(
            {
                "tool_call_id": tool_calls[0].id,
                "role": "tool",
                "name": "fetch_article_chunks_for_rag",
                "content": combined_content,
            }
        )
        second_response = openai_client.chat.completions.create(
            model=openai_model,
            messages=message_chain,
            stream=True
        )
        return second_response
    else:
        return completion.choices[0].message.content


def get_last_update_time(repo_owner, repo_name, file_path):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&per_page=1"
    response = requests.get(url)
    if response.status_code == 200:
        commit_data = response.json()
        if commit_data:
            last_update_time = commit_data[0]['commit']['committer']['date']
            last_update_time = datetime.strptime(last_update_time, "%Y-%m-%dT%H:%M:%SZ")
            formatted_time = last_update_time.strftime("%b %d, %Y")
            return formatted_time
        else:
            return "File not found or no commits for this file."
    else:
        return f"Failed to fetch data: {response.status_code}"


if __name__ == '__main__':
    test_messages = [
        {'role': 'system', 'content': SYSTEM_MESSAGE}
    ]

    print(create_chat_completion_with_rag("Who is Ben Thompson?", test_messages, 'gpt-3.5-turbo'))
    print(create_chat_completion_with_rag("What does Ben think about the Apple Vision Pro?", test_messages, 'gpt-3.5-turbo'))
    print(fetch_article_summaries(["Aggregator's AI Risk", "An Interview with Nat Friedman and Daniel Gross Reasoning About AI"]))
