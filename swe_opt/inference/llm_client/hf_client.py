from huggingface_hub import InferenceClient
from swe_opt.inference.llm_client.base import BaseLLMClient


class HuggingFaceClient(BaseLLMClient):
    """Client for HuggingFace Serverless Inference API (chat completions)."""

    def __init__(
        self,
        model: str,
        api_token: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = 1024,
    ):
        super().__init__(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.client = InferenceClient(model=model, token=api_token)

    def generate(self, prompt: str) -> str:
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content.strip()