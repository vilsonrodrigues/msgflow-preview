from os import getenv
from msgflow.models.providers.vllm import VLLMTextReranker


class _BaseJinaAI:
    """Configurations to use JinaAI models."""
    provider: str = "jinaai"

    def _get_base_url(self):
        base_url = getenv("JINAAI_BASE_URL", "https://api.jina.ai/v1")
        if base_url is None:
            raise ValueError("Please set `JINAAI_BASE_URL`")
        return base_url  
    
    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("JINAAI_API_KEY")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")


class JinaAITextReranker(VLLMTextReranker, _BaseJinaAI):
    """JinaAI Text Reranker."""
