from msgflow.envs import get_env
from msgflow.models.clients.openai import OpenAIChatCompletation


class TogetherChatCompletation(OpenAIChatCompletation):
    r""" 
    
    
    """
    provider: str = "together"

    def _get_base_url(self):
        base_url = get_env("TOGETHER_BASE_URL")
        if base_url is None:
            raise ValueError("Please set `TOGETHER_BASE_URL`")
        return base_url  
    
    def _get_api_key(self):
        keys = get_env("TOGETHER_API_KEY")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")