# src/generation/generator.py

from pathlib import Path
from urllib import response


class Generator:

    def __init__(self, model_config: dict, temperature: float, max_new_tokens: int):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.backend = model_config.get("backend", "huggingface")

        if self.backend == "llamacpp":
            self._init_llamacpp(model_config)
        else:
            self._init_huggingface(model_config.get("model_name", "google/flan-t5-base"))

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def _init_llamacpp(self, model_config: dict):
        from llama_cpp import Llama

        model_path = model_config.get("model_path", "")
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"\n[ERROR] GGUF model not found at: {model_path}\n"
                f"Download Phi-3-mini-4k-instruct-q4.gguf from HuggingFace\n"
                f"and place it at: {model_path}\n"
                f"Or switch models.yaml backend to 'huggingface' to use flan-t5."
            )

        self.llm = Llama(
            model_path=model_path,
            n_ctx=model_config.get("n_ctx", 4096),
            n_threads=model_config.get("n_threads", 4),
            n_gpu_layers=model_config.get("n_gpu_layers", 32), 
            verbose=False,
        )
        print(f"[INFO] Loaded GGUF model: {model_path}")

    def _init_huggingface(self, model_name: str):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
        )

        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_tokenizer.model_max_length = 512

        if "t5" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_seq2seq = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_seq2seq = False

        self.model.eval()
        print(f"[INFO] Loaded HuggingFace model: {model_name}")

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        if self.backend == "llamacpp":
            return self._generate_llamacpp(prompt)
        else:
            return self._generate_huggingface(prompt)
        
    def generate_stream(self, prompt: str):
        """
        Yields string tokens one by one.
        Uses llama-cpp-python's built-in stream=True support.
        """
        stream = self.llm(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.2,
            repeat_penalty=1.1,
            echo=False,
            stop=["<|end|>", "<|user|>", "<|system|>", "\nQuestion:", "\nContext:"],
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def _generate_llamacpp(self, prompt: str) -> str:
        response = self.llm(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repeat_penalty=1.1,
            stop=["<|end|>", "<|user|>", "<|system|>", "\nQuestion:", "\nContext:"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()

    def _generate_huggingface(self, prompt: str) -> str:
        import torch

        inputs = self.hf_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                )

        return self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)