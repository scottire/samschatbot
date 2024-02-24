import chromadb
import requests
import os
import dotenv
import warnings
import urllib.parse
import feedparser
import json
from pprint import pprint
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STRATECHERY_RSS_ID = os.getenv('STRATECHERY_RSS_ID')
STRATECHERY_ACCESS_TOKEN = os.getenv('STRATECHERY_ACCESS_TOKEN')

chroma_client = chromadb.PersistentClient('./chroma.db')


def get_articles_from_rss(rss_feed_url):
    """Returns a list of dictionaries of articles from a given RSS feed showing their url, title, and publish date"""
    feed = feedparser.parse(rss_feed_url)
    articles = []
    for entry in feed.entries:
        article_url = entry.id
        subscriber_url = entry.link
        article_title = entry.title
        article_date = entry.published
        articles.append({'public_url': article_url,
                         'subscriber_url': subscriber_url,
                         'title': article_title,
                         'date': article_date
                         })

    return articles


def save_latest_rss_as_json(rss_feed_url, json_file_name=None):
    """Saves the latest RSS feed as a JSON file"""
    list_of_articles = get_articles_from_rss(rss_feed_url)
    article_json = []
    for article in list_of_articles:
        article_json.append({'title': article['title'],
                             'public_url': article['public_url'],
                             'publish_date': article['date'],
                             'file_location': f'./data/{article["title"]}.md',
                             })
    if json_file_name:
        with open(json_file_name, 'w') as file:
            json.dump(article_json, file)

    return article_json


def get_article_as_markdown(article_url, access_token, article_title=None, save_path=None):
    """Converts a given article url to markdown and save it to the ./data folder"""
    encoded_url = urllib.parse.quote(article_url + f'?access_token={access_token}')
    response = requests.get(f'https://urltomarkdown.herokuapp.com/?url={encoded_url}&title=true')
    if response.status_code == 200:

        pprint(article_title)
        if article_title == 'An Interview with Arm CEO Rene Haas':
            pprint(encoded_url)
            pprint(response.text[:200] + "...")

        article_markdown = response.text

        if article_title is None:
            article_title = urllib.parse.unquote(response.headers['X-Title']).split('–')[0].strip()

        if save_path:
            with open(f'{save_path}/{article_title}.md', 'w') as f:
                f.write(article_markdown)

        return article_markdown
    else:
        print(f"Error: {response.status_code}")
        return


def split_article_into_chunks(article_content, article_title, chunk_size=1000):
    """Splits the given markdown article into digestible chunks and returns them as a list"""
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=0
    )
    chunks = markdown_splitter.create_documents([article_content])
    chunks_with_ids = [{'chunk_id': f"{i}_{article_title}", 'page_content': chunk.page_content} for i, chunk in
                       enumerate(chunks)]
    return chunks_with_ids


def embed_and_save_in_chroma(chunk_id, article_chunk, article_url, article_title, article_date):
    """Embeds and saves a given article chunk to Chroma"""
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_or_create_collection(
        name="stratechery_articles",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )
    collection.upsert(
        ids=[chunk_id],
        documents=[article_chunk],
        metadatas=[{"url": article_url, "title": article_title, "date": article_date}],
    )

    return collection.get("id1", include=["metadatas", "embeddings", "documents"])


def chunk_and_embed_one_article_from_json(json_file_name, article_title):
    """Chunks and embeds the article in the given JSON file to Chroma"""
    with open(json_file_name, 'r') as file:
        articles = json.load(file)

    article = next((article for article in articles if article['title'] == article_title), None)

    with open(article['file_location'], 'r') as file:
        markdown_content = file.read()
    chunks = split_article_into_chunks(markdown_content, article['title'])
    for chunk in chunks:
        print(
            f"({chunk['chunk_id'].split('_')[0]}/{len(chunks) - 1}) {article['title']} - {chunk['page_content'][:50]}...")
        embed_and_save_in_chroma(chunk['chunk_id'],
                                 chunk['page_content'],
                                 article['public_url'],
                                 article['title'],
                                 article['publish_date']
                                 )
    print("Done!")


def chunk_and_embed_articles_from_json(file_name):
    """Chunks and embeds the articles in the given JSON file to Chroma"""
    with open(file_name, 'r') as file:
        articles = json.load(file)

    for article in articles:
        print(f"ARTICLE {articles.index(article)}/{len(articles)} - {article['title']}")
        with open(article['file_location'], 'r') as file:
            markdown_content = file.read()
        chunks = split_article_into_chunks(markdown_content, article['title'])
        for chunk in chunks:
            print(
                f"({chunk['chunk_id'].split('_')[0]}/{len(chunks) - 1}) {article['title']} - {chunk['page_content'][:50]}...")
            embed_and_save_in_chroma(chunk['chunk_id'],
                                     chunk['page_content'],
                                     article['public_url'],
                                     article['title'],
                                     article['publish_date']
                                     )
    print("Done!")


def check_for_latest_articles(rss_feed_url, json_file_name, markdown_save_path, embed=True):
    """Returns a list of new articles that do not exist in the given json file or markdown save path"""
    article_json = save_latest_rss_as_json(rss_feed_url)
    new_articles = []
    for article in article_json:
        if not os.path.exists(f'{markdown_save_path}/{article["title"]}.md'):
            print(f"NEW ARTICLE: {article['title']}")
            get_article_as_markdown(article['public_url'], STRATECHERY_ACCESS_TOKEN, article['title'],
                                    markdown_save_path)
            if embed:
                chunk_and_embed_one_article_from_json(json_file_name, article['title'])
            new_articles.append(article)

    # Read existing data from the file
    with open(json_file_name, 'r') as file:
        all_articles = json.load(file)

    # Append new articles at the beginning of the list
    all_articles[:0] = new_articles

    # Write all the articles back to the json file
    with open(json_file_name, 'w') as file:
        json.dump(all_articles, file, indent=4)

    return new_articles


# save_latest_rss_as_json(f'https://stratechery.passport.online/feed/rss/{STRATECHERY_RSS_ID}',
#                        'data.json')

# chunk_and_embed_articles_from_json('data.json')

print(check_for_latest_articles(f'https://stratechery.passport.online/feed/rss/{STRATECHERY_RSS_ID}',
                                'data.json',
                                './data',
                                embed=True))
