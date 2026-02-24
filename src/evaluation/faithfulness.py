from rouge_score import rouge_scorer as rs


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """
    Measures how much of the answer is grounded in retrieved contexts.
    Uses ROUGE-L F-measure — much more meaningful than word overlap.
    Score closer to 1.0 = answer closely follows the context.
    Score closer to 0.0 = answer diverges from context.
    """
    if not answer.strip() or not contexts:
        return 0.0

    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    combined_context = " ".join(contexts)
    score = scorer.score(combined_context, answer)
    return round(score["rougeL"].fmeasure, 4)