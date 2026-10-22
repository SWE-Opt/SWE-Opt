from google import genai
from google.genai.types import GenerateContentConfig
from swe_opt.inference.llm_client.base import BaseLLMClient

class GeminiClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
        model: str,
        api_version: str,
        temperature=None,
        top_p=None,
        max_tokens=None
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        self.client = genai.Client(
            api_key=api_key,
            http_options={
            "base_url": endpoint_url,
            "api_version": api_version
            }
        )
        
    def generate(self, prompt:str) -> str:

        config_kwargs = {}

        if self.max_tokens is not None:
            config_kwargs["max_output_tokens"] = self.max_tokens
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            config_kwargs["top_p"] = self.top_p

        config = GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config)

        return response.text