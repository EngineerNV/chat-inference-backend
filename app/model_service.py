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

        conversation = ""
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            conversation += f"<{role}>: {content}\n"

        conversation += f"<user>: {message}\n<assistant>:"
        inputs = self.tokenizer(conversation, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "<assistant>:" in full_text:
            return full_text.split("<assistant>:")[-1].strip()
        return full_text.strip()
