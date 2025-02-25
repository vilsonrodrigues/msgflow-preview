from typing import Any, Type, Dict
from msgflow.utils import import_module_from_lib
from msgflow.models.base import BaseClient
from msgflow.models.types import (
    ChatCompletionModel,
    ImageEmbedderModel,
    ImageGenModel,
    TranscriberModel,
    TextEmbedderModel,
    TextRerankerModel,
    TTSModel,    
)

_SUPPORTED_MODEL_TYPES = [
    "chat_completion",
    "audio_classifier",
    "audio_embedder",
    "mask_gen",
    "image_caption",
    "image_classifier",
    "image_embedder",    
    "image_gen",
    "image_segmenter",
    "object_detection",
    "ocr",
    "tts",
    "transcriber",
    "text_classifier",
    "text_embedder",
    "text_reranker",
    "video_classifier",
    "video_gen",    
]

# TODO: _PROVIDER_CONFIG is not so good yet

_PROVIDER_CONFIG = {
    "chat_completion": {
        "providers": ["openai", "google", "amazon"],
        "return_type": ChatCompletionModel
    },
    "image_gen": {
        "providers": ["openai", "google", "amazon"],
        "return_type": ImageGenModel
    },
    "tts": {
        "providers": ["openai", "google", "amazon"],
        "return_type": TTSModel
    },
    "transcriber": {
        "providers": ["openai", "google", "amazon"],
        "return_type": TranscriberModel
    },
    "text_embedder": {
        "providers": ["openai", "google", "amazon", "fast_embedding", "sbert"],
        "return_type": TextEmbedderModel,
        "local_providers": ["sbert"]
    },
    "image_embedder": {
        "providers": ["timm"],
        "return_type": ImageEmbedderModel,
        "local_providers": ["timm"]
    },
    "text_reranker": {
        "providers": ["openai", "sbert"],
        "return_type": TextRerankerModel,
        "local_providers": ["sbert"]
    }
}

_MODEL_NAMESPACE_TRANSLATOR = {
    "openai": "OpenAI",
    "google": "Google",
    "amazon": "Amazon",
    "fast_embedding": "FastEmbedding",
    "diffusers": "Diffusers",
    "timm": "Timm",
    "sbert": "SBERT",
    "ctranslate2": "CTranslate2",
    "tango_flux": "TangoFlux"
}


class Model:

    supported_model_types = _SUPPORTED_MODEL_TYPES
    provider_config = _PROVIDER_CONFIG

    @classmethod
    def _model_path_parser(cls, model_id: str) -> tuple[str, str]:
        provider, model_id = model_id.split("/", 1)
        return provider, model_id

    @classmethod
    def _get_model_class(cls, model_type: str, provider: str) -> Type[BaseClient]:
        if model_type not in cls.provider_config:
            raise ValueError(f"Model type `{model_type}` is not supported")
            
        config = cls.provider_config[model_type]
        if provider not in config["providers"]:
            raise ValueError(f"Provider `{provider}` is not supported for {model_type}")

        provider_class_name = f"{_MODEL_NAMESPACE_TRANSLATOR[provider]}{model_type.title().replace('_', '')}"
        # TODO tem que rever isso
        module_base = "local" if provider in config.get("local_providers", []) else "clients"
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
        instance.deserialize(params)
        return instance

    # Convenience methods for each model type
    @classmethod
    def chat_completion(cls, model_path: str, **kwargs) -> ChatCompletionModel:
        return cls._create_model("chat_completion", model_path, **kwargs)

    @classmethod
    def image_gen(cls, model_path: str, **kwargs) -> ImageGenModel:
        return cls._create_model("image_gen", model_path, **kwargs)

    @classmethod
    def tts(cls, model_path: str, **kwargs) -> TTSModel:
        return cls._create_model("tts", model_path, **kwargs)

    @classmethod
    def transcriber(cls, model_path: str, **kwargs) -> TranscriberModel:
        return cls._create_model("transcriber", model_path, **kwargs)

    @classmethod
    def text_embedder(cls, model_path: str, **kwargs) -> TextEmbedderModel:
        return cls._create_model("text_embedder", model_path, **kwargs)

    @classmethod
    def image_embedder(cls, model_path: str, **kwargs) -> ImageEmbedderModel:
        return cls._create_model("image_embedder", model_path, **kwargs)

    @classmethod
    def text_reranker(cls, model_path: str, **kwargs) -> TextRerankerModel:
        return cls._create_model("text_reranker", model_path, **kwargs)