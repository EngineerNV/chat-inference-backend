import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TinyLlamaService:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def generate_response(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        history = history or []

        messages = [*history, {"role": "user", "content": message}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        do_sample = self.temperature > 0
        generate_kwargs: dict[str, int | bool | float] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature

        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(**inputs, **generate_kwargs)

        new_tokens = output[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
