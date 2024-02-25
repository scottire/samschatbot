# Ben Thompson Stratechery Chatbot

## 🎈 [Use on Streamlit now!](https://unofficial-stratechery-chatbot.streamlit.app/)
![Stratechery Chatbot](Stratechery%20Chatbot%20_%20Streamlit.jpeg)

## What does this bot know?
The bot knows about Ben Thompson, Stratechery, and the [Stratechery](https://stratechery.com/) articles listed in [data.json](data.json). 
The oldest known article dates back to Nov 8, 2023.

## How was this bot built?
- Stratechery articles were saved as markdown files, split into smaller chunks, and embedded in a [Chroma](https://www.trychroma.com/) database.
- On (almost) every query, the bot embeds your query, identifies the 5 most similar article chunks, and places them into GPT's context to answer your question. This technique is known as *[Retreival-Augmented Generation (RAG)](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)*.
- RAG is far from perfect! I used the open-sourced [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model to create the embeddings. I used it because it's free (I'm cheap) and has [good speed and performance](https://huggingface.co/blog/mteb) for what you're getting.
- I removed all those markdown Stratechery articles from this repo out of respect to Ben Thompson.

### [data.py](data.py)
data.py is a mess of a codebase that shows how I retreieved, chunked, and embedded the Straterchery articles

### [chatbot.py](chatbot.py)
chatbot.py is the UI logic for the Streamlit chatbot.

### [chatbot_helper.py](chatbot_helper.py)
chatbot_helper.py is the helper functions for the Streamlit chatbot. This is where the magic happens with the GPT chat completions.

## Who built this bot?
This bot was built by [Ben Wallace](https://twitter.com/DJbennyBuff). He's been a Stratechery subscriber for about 4 years. He wanted to build a chatbot from scratch and was inspired by the [LennyBot](https://www.lennybot.com/), a GPT bot trained on Lenny's Newsletters.

_Disclaimer: This app is not affiliated with, endorsed by, or approved by Ben Thompson or Stratechery._