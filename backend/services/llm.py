import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def ask_llm(context, question):

    prompt = f"""
You are an AI Medical Research Assistant.

Answer ONLY using the research papers below.

If the answer is not contained in the papers, say:
"I don't have enough evidence from the retrieved papers."

Research Papers:

{context}

Question:
{question}

Provide:
1. Clear answer
2. Summary
3. Mention evidence
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",   # or another compatible model available to your API
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content