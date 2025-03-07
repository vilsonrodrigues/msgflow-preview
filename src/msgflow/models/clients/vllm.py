from msgflow.envs import get_env
from msgflow.models.clients.openai import OpenAIChatCompletation


class VLLMChatCompletation(OpenAIChatCompletation):
    r""" 
    
    
    """
    provider: str = "vllm"

    def _get_base_url(self):
        base_url = get_env("VLLM_BASE_URL")
        if base_url is None:
            raise ValueError("Please set `VLLM_BASE_URL`")
        return base_url  
    
    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = get_env("VLLM_API_KEY", "vllm")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")