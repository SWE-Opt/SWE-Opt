import json
import requests
from swe_opt.inference.llm_client.base import BaseLLMClient

class OllamaClient(BaseLLMClient):
    """Client for local Ollama inference server."""

    def __init__(
        self,
        model: str,
        endpoint: str = "http://localhost:11434/api/generate",
        temperature: float | None = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
    ):
        super().__init__(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.endpoint = endpoint
        self.thinking = thinking

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "think": self.thinking,
            "options": {},
        }

        if self.temperature is not None:
            payload["options"]["temperature"] = self.temperature

        if self.top_p is not None:
            payload["options"]["top_p"] = self.top_p

        if self.max_tokens is not None:
            payload["options"]["num_predict"] = self.max_tokens

        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()

        lines = response.text.strip().splitlines()
        full_text = ""
        for line in lines:
            try:
                chunk = json.loads(line)
                if chunk.get("response"):
                    full_text += chunk["response"]
            except Exception:
                continue

        return full_text.strip()
