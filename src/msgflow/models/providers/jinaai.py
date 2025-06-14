from os import getenv
from typing import List, Optional, Union
from msgflow.models.providers.openai import HTTPXModelClient
from msgflow.models.providers.vllm import VLLMTextReranker
from msgflow.models.response import ModelResponse
from msgflow.models.types import (
    ImageClassifierModel,
    ImageEmbedderModel,
    TextClassifierModel,
    TextEmbedderModel,
)    


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


class JinaAITextEmbedder(TextEmbedderModel, HTTPXModelClient, _BaseJinaAI):
    """JinaAI Text Embedder."""
    url_path: str = "/embeddings"
    batch_support = True

    def __init__(
        self,
        *,
        model_id: str,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"dimensions": dimensions}
        self._initialize()
        self._get_api_key()

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        embedding = [item["embedding"] for item in data]
        if len(embedding) == 1: # Compatibility
            embedding = embedding[0]
        response.add(embedding)
        return response

    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        if isinstance(data, str):
            data = [data]
        inputs = [{"text": item} for item in data]
        response = self._generate(input=inputs)
        return response


class JinaAIImageEmbedder(ImageEmbedderModel, JinaAITextEmbedder):
    """JinaAI Image Embedder."""
    url_path: str = "/embeddings"
    batch_support = True

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_embedding")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        embedding = [item["embedding"] for item in data]
        if len(embedding) == 1: # Compatibility
            embedding = embedding[0]
        response.add(embedding)
        return response

    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        if isinstance(data, str):
            data = [data]
        inputs = [{"image": item} for item in data]
        response = self._generate(input=inputs)
        return response

class JinaAITextClassifier(TextClassifierModel, HTTPXModelClient, _BaseJinaAI):
    """JinaAI Text Classifier."""
    url_path: str = "/classify"
    batch_support = True
    
    def __init__(
        self, 
        model_id, 
        labels: List[str], 
        base_url: Optional[str] = None
    ):
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"labels": labels}
        self._initialize()
        self._get_api_key()

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_classification")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        pred = [{"label": item["prediction"], "score": item["score"]} for item in data]
        if len(pred) == 1: # Compatibility
            pred = pred[0]
        response.add(pred)
        return response

    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        if isinstance(data, str):
            data = [data]
        inputs = [{"text": item} for item in data]
        response = self._generate(input=inputs)
        return response


class JinaAIImageClassifier(JinaAITextClassifier, ImageClassifierModel):
    """JinaAI Image Classifier."""

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_classification")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        pred = [{"label": item["prediction"], "score": item["score"]} for item in data]
        if len(pred) == 1: # Compatibility
            pred = pred[0]
        response.add(pred)
        return response

    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        if isinstance(data, str):
            data = [data]
        inputs = [{"image": item} for item in data]
        response = self._generate(input=inputs)
        return response
