

from openai import OpenAI
from swe_opt.inference.llm_client.base import BaseLLMClient

# This client uses the OpenAI Responses API, which is a more recent API that supports a wider range of models and features. 
# It is recommended to use this client for new implementations
# While the OpenAIChatCompletionsClient can be used for compatibility with older code that relies on the older Chat Completions API.

class OpenAIResponsesClient(BaseLLMClient):
    """Client for OpenAI models using the Responses API."""

    def __init__(self, 
                 api_key: str, 
                 endpoint_url: str,
                 model: str, 
                 temperature=None, 
                 top_p=None, 
                 max_tokens=None):
        
        super().__init__(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.client = OpenAI(
            base_url=endpoint_url,
            api_key=api_key)

    def generate(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "input": prompt,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens

        response = self.client.responses.create(**kwargs)
        return response.output_text.strip()

