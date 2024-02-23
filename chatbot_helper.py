import chromadb
import dotenv
import os
import json
from openai import OpenAI

dotenv.load_dotenv()

if os.getenv('IS_LOCAL') != 'true':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI()

chroma_client = chromadb.PersistentClient('./chroma.db')


def query_articles(query_text):
    q = chroma_client.get_collection("stratechery_articles").query(query_texts=query_text, n_results=5)
    return q


def fetch_article_chunks_for_rag(query_text):
    """Returns a clean string of the query text and the top 5 results from a given query"""
    q = query_articles(query_text)

    documents = q['documents'][0]
    distances = q['distances'][0]
    metadatas = q['metadatas'][0]
    ids = q['ids'][0]

    # Combine all relevant information into a single list
    combined = list(zip(distances, documents, metadatas, ids))

    # Sort by distance in descending order
    combined.sort(key=lambda x: x[0], reverse=True)

    # Group by article title
    grouped = {}
    for distance, document, metadata, article_id in combined:
        article_title = metadata['title']
        if article_title not in grouped:
            grouped[article_title] = {'url': metadata['url'], 'documents': []}
        grouped[article_title]['documents'].append(document)

    # Build the final string
    result = []
    for title, info in grouped.items():
        result.append(f"[{title}]({info['url']})")
        result.extend(info['documents'])
        result.append("-----")
    return "\n".join(result).strip("-----\n")


def get_articles_info_from_json(file_name):
    with open(file_name, 'r') as file:
        articles = json.load(file)
    article_titles = [article['title'] for article in articles]
    return len(articles), article_titles


num_articles, article_titles = get_articles_info_from_json('data.json')


SYSTEM_MESSAGE = f"""* You are a bot that knows everything about Ben Thompson's Stratechery articles (https://stratechery.com/). You are smart, witty, and love tech! You talk candidly and casually.
* You are trained on Stratechery articles since Nov 6, 2023. Ben Wallace (https://ben-wallace.replit.app/) created you. Your code can be found at https://github.com/benfwalla/BenThompsonChatbot. You are not approved by Ben Thompson.
* You are trained on the {num_articles} most recent Stratechery articles. The oldest article is Nov 6, 2023. Here are their names in descending order: {article_titles}
* You will answer questions using Stratechery articles. You will always refer to the specific name of the article you are citing and hyperlink to its url, as such: [Article Title](Article URL).
* If you are referring to Ben Thompson, just say "Ben". If you can't answer, you will explain why and suggest sending the question to email@sharptech.fm where Ben can answer it directly!
* A user's questions may be followed by a bunch of possible answers from Stratechery articles. Each article is is separated by `-----` and is formatted as such: `[Article Title](Article URL)\\n[Chunk of Article Content]`. Use your best judgement to answer the user's query based on the articles provided.
* Facts about Stratechery: Stratechery provides analysis on the business, strategy, and impact of technology and media, and how technology is changing society. It's known for its weekly articles, which are free, and three subscriber-only Daily Updates per week. The site is recommended by The New York Times and has subscribers from over 85 countries, including executives, venture capitalists, and tech enthusiasts. The Stratechery Plus bundle includes the Stratechery Update, Stratechery Interviews, Dithering podcast, Sharp Tech podcast, Sharp China podcast, and Greatest Of All Talk podcast, available for $12 per month, or $120 per year.
* Facts about Ben Thompson: Ben Thompson is the author of Stratechery. He's based in Taipei, Taiwan, and has worked at companies like Apple (interned), Microsoft, and Automattic. He holds an MBA from Kellogg School of Management and an MEM from McCormick Engineering school. Ben has been writing Stratechery since 2013, and it has been his full-time job since 2014. You can follow him on X @benthompson (https://x.com/benthompson)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_article_chunks_for_rag",
            "description": "This function a chunk of text from Stratechery articles that are relevant to the query. "
                           "ONLY use this function if the existing information in your message history is not enough.",
        }
    }
]


def create_chat_completion_with_rag(query_text, message_chain, openai_model):
    message_chain.append({"role": "user", "content": query_text})

    completion = openai_client.chat.completions.create(
        model=openai_model,
        messages=message_chain,
        tools=TOOLS
    )
    response_message = completion.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls and tool_calls[0].function.name == "fetch_article_chunks_for_rag":
        print("RAG is needed!")
        article_chunks = fetch_article_chunks_for_rag(query_text)
        message_chain.append(response_message)
        message_chain.append(
            {
                "tool_call_id": tool_calls[0].id,
                "role": "tool",
                "name": "fetch_article_chunks_for_rag",
                "content": article_chunks,
            }
        )
        second_response = openai_client.chat.completions.create(
            model=openai_model,
            messages=message_chain,
            stream=True
        )
        return second_response
    else:
        print("RAG is not needed!")
        return completion.choices[0].message.content


# test_messages = [
#     {'role': 'system', 'content': SYSTEM_MESSAGE}
# ]

#print(create_chat_completion_with_rag("Who is Ben Thompson?", test_messages, 'gpt-3.5-turbo'))
#print(create_chat_completion_with_rag("What does Ben think about the Apple Vision Pro?", test_messages, 'gpt-3.5-turbo'))
