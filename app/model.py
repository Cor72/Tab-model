from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B"


class CompletionModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
        )
        self.model.eval()

    def complete(self, prefix: str, max_new_tokens: int = 24) -> str:

        messages = [
            {
                "role": "user",
                "content": (
                    "请接着下面的前缀续写，只补完当前句子。"
                    "只输出新增内容，不重复前缀，不解释。\n"
                    f"前缀：{prefix}"
                ),
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                num_beams=1,
                no_repeat_ngram_size=3,
                penalty_alpha=0.6,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
