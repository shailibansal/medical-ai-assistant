from ollama import chat


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

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]