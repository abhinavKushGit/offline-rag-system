def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""<|system|>
You are a helpful assistant. Answer the question using only the context below. Be concise and direct if asked about being concise and direct. If the answer is not in the context, say "I don't know."<|end|>
<|user|>
Context:
{context}

Question: {query}<|end|>
<|assistant|>
"""
    return prompt