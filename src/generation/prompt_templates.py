def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""<|system|>
You are a helpful assistant. The context below contains retrieved content from the uploaded file — this may include audio transcripts, video captions, image descriptions, or document text. Answer the question using only the context. Be concise. If the answer is not in the context, say "I don't know."<|end|>
<|user|>
Context:
{context}

Question: {query}<|end|>
<|assistant|>
"""
    return prompt