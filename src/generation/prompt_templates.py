def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""<|system|>
You are a helpful assistant analyzing multimedia content. Answer the question using ONLY the context provided.
The context contains descriptions of video frames with timestamps, audio transcripts, image captions, or document text.
When asked about a video, use the video frame descriptions and transcripts in the context.
If the answer is not in the context, say "I don't know."
Do not repeat yourself. Give one clear, concise answer and stop.<|end|>
<|user|>
Context:
{context}

Question: {query}<|end|>
<|assistant|>
"""
    return prompt