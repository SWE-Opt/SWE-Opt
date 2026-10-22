from anthropic import Anthropic
from swe_opt.inference.llm_client.base import BaseLLMClient

class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
        model: str,
        temperature=None,
        top_p=None,
        max_tokens=None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        self.client = Anthropic(
            base_url=endpoint_url,
            api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "thinking": {"type": "disabled"},
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        response = self.client.messages.create(**kwargs)

        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)

        return "".join(parts).strip()