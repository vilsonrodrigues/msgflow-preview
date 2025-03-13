from typing import Any, Dict, Type
from msgflow.models.base import BaseClient
from msgflow.models.types import (
    ASRModel,
    ChatCompletionModel,
    ImageTextToImageModel,
    ImageEmbedderModel,
    TextEmbedderModel,
    TextRerankerModel,
    TTSModel,    
)
from msgflow.utils.imports import import_module_from_lib


_SUPPORTED_MODEL_TYPES = [
    "asr",    
    "chat_completion",
    #"audio_classifier",
    #"audio_embedder",
    #"mask_gen",
    #"image_caption",
    #"image_classifier",
    "image_embedder",    
    #"image_gen",
    #"image_segmenter",
    "image_text_to_image",
    #"object_detection",
    #"ocr",
    "tts",
    #"text_classifier",
    "text_embedder",
    #"text_reranker",
    #"video_classifier",
    #"video_gen",    
]

_MODEL_NAMESPACE_TRANSLATOR = {
    "openai": "OpenAI", 
    #"google": "Google", 
    #"amazon": "Amazon", 
    #"fast_embedding": "FastEmbedding",
    "timm": "TIMM",
    #"sbert": "SBERT"
    "local_vllm": "LocalVLLM",
    "vllm": "VLLM",
    "together": "Together"
} 

_CHAT_COMPLETION_PROVIDERS = ["openai", "local_vllm", "vllm", "together"] 
_IMAGE_EMBEDDER_PROVIDERS = ["timm"]
_IMAGE_TEXT_TO_IMAGE_PROVIDERS = ["openai"]
_TTS_PROVIDERS = ["openai"]
_ASR_PROVIDERS = ["openai"]
_TEXT_EMBEDDER_PROVIDERS = ["openai"]
#_TEXT_RERANKER_PROVIDERS = ["openai", "sbert"]

_PROVIDERS_BY_MODEL_TYPE = {
    "chat_completion": _CHAT_COMPLETION_PROVIDERS,
    "asr": _ASR_PROVIDERS,
    "image_embedder": _IMAGE_EMBEDDER_PROVIDERS,
    "image_text_to_image": _IMAGE_TEXT_TO_IMAGE_PROVIDERS,
    "tts": _TTS_PROVIDERS,
    "text_embedder": _TEXT_EMBEDDER_PROVIDERS,    
}

_LOCAL_PROVIDERS = ["local_vllm"]

class Model:
    supported_model_types = _SUPPORTED_MODEL_TYPES
    providers_by_model_type = _PROVIDERS_BY_MODEL_TYPE
    local_providers = _LOCAL_PROVIDERS

    @classmethod
    def _model_path_parser(cls, model_id: str) -> tuple[str, str]:
        provider, model_id = model_id.split("/", 1)
        return provider, model_id

    @classmethod
    def _get_model_class(cls, model_type: str, provider: str) -> Type[BaseClient]:
        if model_type not in cls.supported_model_types:
            raise ValueError(f"Model type `{model_type}` is not supported")
            
        providers = cls.providers_by_model_type[model_type]
        if provider not in providers:
            raise ValueError(f"Provider `{provider}` is not supported for {model_type}")

        provider_class_name = f"{_MODEL_NAMESPACE_TRANSLATOR[provider]}{model_type.title().replace('_', '')}"
                
        module_base = "local" if provider in cls.local_providers else "clients"
        
        # Solve cases as "local-vllm" to "vllm" to py files
        provider = provider.replace("local_", "")
        module_name = f"msgflow.models.{module_base}.{provider}"
                
        return import_module_from_lib(provider_class_name, module_name)

    @classmethod
    def _create_model(cls, model_type: str, model_path: str, **kwargs) -> BaseClient:
        provider, model_id = cls._model_path_parser(model_path)
        model_cls = cls._get_model_class(model_type, provider)
        return model_cls(model_id=model_id, **kwargs)

    @classmethod
    def from_serialized(cls, provider: str, model_type: str, params: Dict[str, Any]) -> BaseClient:
        """
        Creates a model instance from serialized parameters without calling __init__.
        
        Args:
            provider: The model provider (e.g., "openai", "google")
            model_type: The type of model (e.g., "chat_completation", "text_embedder")
            params: Dictionary containing the serialized model parameters
            
        Returns:
            An instance of the appropriate model class with restored state
        """
        model_cls = cls._get_model_class(model_type, provider)
        # Create instance without calling __init__
        instance = object.__new__(model_cls)
        # Restore the instance state
        instance.from_serialized(params)
        return instance

    @classmethod
    def chat_completion(cls, model_path: str, **kwargs) -> ChatCompletionModel:
        return cls._create_model("chat_completion", model_path, **kwargs)

    @classmethod
    def image_text_to_image(cls, model_path: str, **kwargs) -> ImageTextToImageModel:
        return cls._create_model("image_gen", model_path, **kwargs)

    @classmethod
    def tts(cls, model_path: str, **kwargs) -> TTSModel:
        return cls._create_model("tts", model_path, **kwargs)

    @classmethod
    def asr(cls, model_path: str, **kwargs) -> ASRModel:
        return cls._create_model("asr", model_path, **kwargs)

    @classmethod
    def text_embedder(cls, model_path: str, **kwargs) -> TextEmbedderModel:
        return cls._create_model("text_embedder", model_path, **kwargs)

    @classmethod
    def image_embedder(cls, model_path: str, **kwargs) -> ImageEmbedderModel:
        return cls._create_model("image_embedder", model_path, **kwargs)

    @classmethod
    def text_reranker(cls, model_path: str, **kwargs) -> TextRerankerModel:
        return cls._create_model("text_reranker", model_path, **kwargs)