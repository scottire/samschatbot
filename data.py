import chromadb
import requests
import os
import dotenv
import warnings
import urllib.parse
import feedparser
from pprint import pprint
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
STRATECHERY_RSS_ID = os.getenv('STRATECHERY_RSS_ID')

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
        articles.append({'public_url': article_url, 'subscriber_url': subscriber_url, 'title': article_title, 'date': article_date})

    return articles


def get_article_as_markdown(article_url, save_path=None):
    """Converts a given article url to markdown and save it to the ./data folder"""
    response = requests.get(f'https://urltomarkdown.herokuapp.com/?url={article_url}&title=true')
    if response.status_code == 200:

        article_title = urllib.parse.unquote(response.headers['X-Title']).split('–')[0].strip()
        article_markdown = response.text

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
    chunks_with_ids = [{'chunk_id': f"{i}_{article_title}", 'page_content': chunk.page_content} for i, chunk in enumerate(chunks)]
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


def embed_and_save_latest_rss():
    list_of_articles = get_articles_from_rss(f'https://stratechery.passport.online/feed/rss/{STRATECHERY_RSS_ID}')
    for article in list_of_articles:
        print(f"Chunking {article['title']}")

        list_of_chunks = split_article_into_chunks(
            get_article_as_markdown(article['subscriber_url'], './data'),
            article['title']
        )

        for article_chunk in list_of_chunks:
            print(f"   Embedding and Saving {article_chunk['chunk_id']}")
            embed_and_save_in_chroma(
                article_chunk['chunk_id'],
                article_chunk['page_content'],
                article['public_url'],
                article['title'],
                article['date']
            )

