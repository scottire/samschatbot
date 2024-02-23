import random
import time
import streamlit as st
from chatbot_helper import SYSTEM_MESSAGE, create_chat_completion_with_rag
from openai import OpenAI


with open("styles/styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.title("Ben Thompson's Stratechery Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

# SIDEBAR
openai_client = OpenAI()
with st.sidebar:
    gpt_model = st.selectbox('Select a Model',
                             ('gpt-3.5-turbo', 'gpt-4-turbo-preview')
                             )

# CHATBOT
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input(placeholder="What does Ben think about the Apple Vision Pro?"):
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
