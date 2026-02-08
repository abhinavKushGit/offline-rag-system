import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)


class Generator:
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_tokens  # 🔑 output-only limit

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = 512  # 🔒 hard safety

        if "t5" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_seq2seq = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_seq2seq = False

        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
