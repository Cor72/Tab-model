from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B-Base"


class CompletionModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
        )
        self.model.eval()

    def complete(self, prefix: str, max_new_tokens: int = 24) -> str:
        # Base 模型直接接收需要续写的原文。
        inputs = self.tokenizer(prefix, return_tensors="pt")

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
            )

        # 只返回新增的文字，不包含输入前缀。
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
