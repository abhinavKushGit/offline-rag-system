def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question ONLY using the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()
