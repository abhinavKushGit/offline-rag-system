from transformers import AutoTokenizer


class TokenBudgeter:
    def __init__(self, model_name: str, max_tokens: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def select_contexts(self, contexts: list[str], question: str) -> list[str]:
        selected = []
        total_tokens = len(self.tokenizer.encode(question))

        for ctx in contexts:
            ctx_tokens = len(self.tokenizer.encode(ctx))
            if total_tokens + ctx_tokens > self.max_tokens:
                break
            selected.append(ctx)
            total_tokens += ctx_tokens

        return selected
