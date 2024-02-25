import random
import time
import streamlit as st
from chatbot_helper import SYSTEM_MESSAGE, create_chat_completion_with_rag, num_articles, get_last_update_time
from openai import OpenAI

st.set_page_config(
    page_title="Stratechery Chatbot",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("styles/styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

# SIDEBAR
LATEST_DATA_UPDATE = get_last_update_time("benfwalla", "BenThompsonChatbot", "data.json")
APP_DESCRIPTION = f"""
- It is *not* approved by Ben Thompson or any Stratechery affiliates.
"""

openai_client = OpenAI()
with st.sidebar:
    gpt_model = st.selectbox('Select a Model', ('gpt-3.5-turbo', 'gpt-4-turbo-preview'))
    st.divider()
    with st.expander("What does this bot know?"):
        st.write(f"The bot knows about Ben Thompson, Stratechery, and the {num_articles} most recent [Stratechery](https://stratechery.com/) articles. The oldest known article dates back to Nov 8, 2023. **It was last updated on {LATEST_DATA_UPDATE}.**")
    with st.expander("How was this bot built?"):
        st.write(f"""
        - Stratechery articles were saved as markdown files, split into smaller chunks, and embedded in a [Chroma](https://www.trychroma.com/) database.
        - On (almost) every query, the bot embeds your query, identifies the 5 most similar article chunks, and places them into GPT's context to answer your question. This technique is known as *[Retreival-Augmented Generation (RAG)](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)*.
        - RAG is far from perfect! I used the open-sourced [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model to create the embeddings. I used it because it's free (I'm cheap) and has [good speed and performance](https://huggingface.co/blog/mteb) for what you're getting.
        - You can find the code **[here](https://github.com/benfwalla/BenThompsonChatbot)**. I removed all those markdown Stratechery articles from the repo out of respect to Ben Thompson.
        """)
    with st.expander("Who built this bot?"):
        st.write("This bot was built by [Ben Wallace](https://twitter.com/DJbennyBuff). He's been a Stratechery subscriber for about 4 years. He wanted to build a chatbot from scratch and was inspired by the [LennyBot](https://www.lennybot.com/), a GPT bot trained on Lenny's Newsletters.")
    st.divider()
    st.caption("_Disclaimer: This app is not affiliated with, endorsed by, or approved by Ben Thompson or Stratechery._")
    st.caption(f"Last updated: {LATEST_DATA_UPDATE}")

# CHATBOT
st.title("Ben Thompson Stratechery Chatbot")
st.caption(f"_Ask me anything about Stratechery! I'm trained on the {num_articles} most recent Stratechery articles._")


def add_message_and_respond(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(""):
            stream = create_chat_completion_with_rag(prompt,
                                                     [{"role": msg["role"], "content": msg["content"]} for msg in
                                                      st.session_state.messages],
                                                     gpt_model)
        if type(stream) is str:
            def stream_data():
                for word in stream.split():
                    yield word + " "
                    time.sleep(random.uniform(0.01, 0.03))

            st.write_stream(stream_data())
            response = stream
        else:
            response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


button_container = st.empty()
button_string = ""
with button_container:
    col1, col2 = st.columns(2, gap="small")
    questions = [
        "What does Ben think of the Vision Pro?",
        "Provide the key points in Stratechery's analysis of Microsoft's acquisition of Activision Blizzard",
        "What is Disney's strategy moving forward?",
        "Which articles talk about Substack?"
    ]
    with col1:
        if st.button(questions[0], use_container_width=True):
            button_string = questions[0]
        if st.button(questions[1], use_container_width=True):
            button_string = questions[1]
    with col2:
        if st.button(questions[2], use_container_width=True):
            button_string = questions[2]
        if st.button(questions[3], use_container_width=True):
            button_string = questions[3]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if button_string:
    button_container.empty()
    add_message_and_respond(button_string)

if prompt := st.chat_input(placeholder="Message StratecheryBot..."):
    add_message_and_respond(prompt)
