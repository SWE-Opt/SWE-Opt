import os
from dotenv import load_dotenv

def get_llm_client(provider: str, **kwargs):
    """
    Factory function to instantiate the correct LLM client.
    Supported providers: openai_chat, openai_responses, ollama, hf, anthropic, gemini, nvidia_nim
    """
    provider = provider.lower()

    if provider == "nvidia_nim":
        from swe_opt.inference.llm_client.nvidia_nim_client import NvidiaNIMClient
        return NvidiaNIMClient(
            api_key=_load_key_from_env(provider),
            model=kwargs["model"],
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            thinking=kwargs.get("thinking", False),
        )

    if provider == "zhipu":
        from swe_opt.inference.llm_client.zhipu_client import ZhipuClient
        return ZhipuClient(
            api_key=_load_key_from_env(provider),
            model=kwargs["model"],
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            thinking=kwargs.get("thinking", False),
        )

    if provider == "minimax":
        from swe_opt.inference.llm_client.nvidia_nim_openai_client import NvidiaNIMOpenAIClient
        return NvidiaNIMOpenAIClient(
            api_key=_load_key_from_env("nvidia_nim"),
            model=kwargs["model"],
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            thinking_kwargs=None,
        )

    if provider == "ollama":
        from swe_opt.inference.llm_client.ollama_client import OllamaClient
        return OllamaClient(
            model=kwargs["model"],
            endpoint="http://localhost:11434/api/generate",
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
        )
    elif provider == "hf":
        from swe_opt.inference.llm_client.hf_client import HuggingFaceClient
        return HuggingFaceClient(
            model=kwargs["model"],
            api_token=_load_key_from_env(provider),
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=kwargs.get("max_tokens", 1024),
        )

    kwargs["api_key"] = _load_key_from_env(provider)

    if provider == "openai_responses":
        from swe_opt.inference.llm_client.openai_responses_client import OpenAIResponsesClient
        return OpenAIResponsesClient(**kwargs)
    
    elif provider == "openai_chat":
        from swe_opt.inference.llm_client.openai_chat_client import OpenAIChatCompletionsClient
        return OpenAIChatCompletionsClient(**kwargs)
    
    elif provider == "anthropic":
        from swe_opt.inference.llm_client.anthropic_client import AnthropicClient
        return AnthropicClient(**kwargs)
    
    elif provider == "gemini":
        from swe_opt.inference.llm_client.gemini_client import GeminiClient
        return GeminiClient(**kwargs)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def _load_key_from_env(provider: str):
    """
    Load API key from environment or .env file based on provider name.

    Supported env vars:
      - OPENAI_API_KEY
      - HUGGINGFACE_API_KEY
      - OLLAMA_API_KEY
      - ANTHROPIC_API_KEY
      - GEMINI_API_KEY
    """

    load_dotenv() 
    env_map = {
        "openai_chat": "OPENAI_API_KEY",
        "openai_responses": "OPENAI_API_KEY",
        "hf": "HUGGINGFACE_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "nvidia_nim": "NVIDIA_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
    }
    key_var = env_map.get(provider.lower())
    if not key_var:
        raise ValueError(f"Unsupported provider for key lookup: {provider}")

    key = os.getenv(key_var)
    if not key:
        raise EnvironmentError(
            f"Missing API key for '{provider}'. Expected environment variable: {key_var}"
        )
    return key
