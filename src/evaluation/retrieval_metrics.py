def recall_at_k(retrieved_chunks, relevant_phrases, k):
    """
    Recall@k using phrase-level relevance.
    """
    retrieved_k = retrieved_chunks[:k]

    hits = 0
    for phrase in relevant_phrases:
        for chunk in retrieved_k:
            if phrase.lower() in chunk.lower():
                hits += 1
                break

    if not relevant_phrases:
        return 0.0

    return hits / len(relevant_phrases)
