def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""<|system|>
You are a helpful assistant. Answer the question using ONLY the context provided. 
If the answer is not in the context, say "I don't know."
Do not repeat yourself. Give one clear, concise answer and stop.<|end|>
<|user|>
Context:
{context}

Question: {query}<|end|>
<|assistant|>
"""
    return prompt