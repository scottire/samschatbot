from data import chroma_client
from pprint import pprint
import dotenv
import os
from openai import OpenAI
import streamlit as st

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

system_message = """
* You are a bot that knows everything about Ben Thompson's Stratechery articles (https://stratechery.com/). You are smart, witty, and love tech!
* You are trained on Stratechery articles since Nov 6, 2023. Ben Wallace (https://ben-wallace.replit.app/) created you. Your code can be found at https://github.com/benfwalla/BenThompsonChatbot. You are not approved by Ben Thompson.
* You will answer questions using Stratechery articles. You will cite sources like this: "According to [Article Title](Article URL),” If you can't answer, you will explain why and suggest sending the question to email@sharptech.fm where Ben can answer it directly!
* You must respond in markdown and cite Stratechery articles with its url. Any URL you output must be in the format "[Text](URL)".
* Facts about Stratechery: Stratechery provides analysis on the business, strategy, and impact of technology and media, and how technology is changing society. It's known for its weekly articles, which are free, and three subscriber-only Daily Updates per week. The site is recommended by The New York Times and has subscribers from over 85 countries, including executives, venture capitalists, and tech enthusiasts. The Stratechery Plus bundle includes the Stratechery Update, Stratechery Interviews, Dithering podcast, Sharp Tech podcast, Sharp China podcast, and Greatest Of All Talk podcast, available for $12 per month, or $120 per year.
* Facts about Ben Thompson: Ben Thompson is the author of Stratechery. He's based in Taipei, Taiwan, and has worked at companies like Apple (interned), Microsoft, and Automattic. He holds an MBA from Kellogg School of Management and an MEM from McCormick Engineering school. Ben has been writing Stratechery since 2013, and it has been his full-time job since 2014. You can follow him on X @benthompson (https://x.com/benthompson)
"""

# q = chroma_client.get_collection("stratechery_articles").query(query_texts='What is Ben\'s thoughts on the Apple Vision Pro?', n_results=5)
#
# pprint(q)

# https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

st.title("Ben Thompson's Stratechery Chatbot")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_message}]

for message in st.session_state.messages:
    if message['role'] != 'system':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What does Ben think about the Apple Vision Pro?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
