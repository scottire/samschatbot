from langchain.text_splitter import MarkdownHeaderTextSplitter
from openai import OpenAI
from pprint import pprint
import dotenv

dotenv.load_dotenv()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4")
]


def summarize_article(markdown_file):
    with open(markdown_file, 'r') as f:
        markdown_content = f.read()

    if "* * *" in markdown_content:
        markdown_content = ''.join(markdown_content.split("* * *")[:-1])

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    section_splits = markdown_splitter.split_text(markdown_content)

    openai_client = OpenAI()

    article_type = "interview" if "Interview" in markdown_file else "article"

    section_summaries = []
    for section in section_splits:
        section_content = section.page_content
        section_header = list(section.metadata.values())[-1]
        section_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Summarize the following snippet of a Stratechery {article_type} in markdown. "
                                              f"Use 'Ben' when describing the author and his points in the third person. Do not use descriptive clauses like 'in the interview,'. Be concise and objective. Return plain text, no formatting: {section_content}"}
            ]
        )
        section_summary = section_header + ": " + section_completion.choices[0].message.content
        section_summaries.append(section_summary)
        print(section_summary)

    print("\n\n")
    section_summaries = '\n'.join(section_summaries)
    article_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"The following is a set of summaries from a Stratechery article: {section_summaries}"
                                          f"Take these and distill it in a final summary about the article. In the event of an interview, intuit what the name abbreviations are from the section headers and use their names. Be concise and objective. Return plain text, no formatting."}
        ]
    )
    return article_completion.choices[0].message.content


print(summarize_article('./data/An Interview with Arm CEO Rene Haas.md'))
