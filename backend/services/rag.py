from services.vectordb import search_papers
from services.llm import ask_llm


def answer_question(question):

    papers = search_papers(question)

    context = ""

    for paper in papers:

        context += f"""
Title:
{paper['title']}

Journal:
{paper['journal']}

Content:
{paper['content']}

------------------------------------
"""

    answer = ask_llm(context, question)

    return {
        "answer": answer,
        "sources": papers
    }