from msgflow.data.retrievers.types import HybridRetriever, LexicalRetriever, SemanticRetriever
from msgflow.utils.imports import import_module_from_lib


_LEXICAL_RETRIEVER_PROVIDERS = ["bm25"]
_SEMANTIC_RETRIEVER_PROVIDERS = []
_HYBRID_RETRIEVER_PROVIDERS = []

_RETRIEVER_NAMESPACE_TRANSLATOR = {
    "bm25": "BM25",
}


class Retriever:
    lexical_retriever_providers = _LEXICAL_RETRIEVER_PROVIDERS
    semantic_retriever_providers = _SEMANTIC_RETRIEVER_PROVIDERS
    hybrid_retriever_providers = _HYBRID_RETRIEVER_PROVIDERS

    @classmethod
    def lexical(cls, provider: str, **kwargs) -> LexicalRetriever:
        if provider not in cls.lexical_retriever_providers:
            raise ValueError(
                f"Provider `{provider}` is not supported for Lexical Retriever"
            )
        provider_class_name = (
            f"{_RETRIEVER_NAMESPACE_TRANSLATOR[provider]}LexicalRetriever"
        )
        module_name = f"msgflow.data.retrievers.{provider}"
        lexcial_cls = import_module_from_lib(provider_class_name, module_name)
        retriever = lexcial_cls(**kwargs)
        return retriever

    @classmethod
    def semantic(cls, provider: str, **kwargs) -> SemanticRetriever:
        if provider not in cls.semantic_retriever_providers:
            raise ValueError(
                f"Provider `{provider}` is not supported for Semantic Retriever"
            )
        provider_class_name = (
            f"{_RETRIEVER_NAMESPACE_TRANSLATOR[provider]}SemanticRetriever"
        )
        module_name = f"msgflow.data.retrivers.{provider}"
        semantic_cls = import_module_from_lib(provider_class_name, module_name)
        retriever = semantic_cls(**kwargs)
        return retriever

    @classmethod
    def hybrid(cls, provider: str, **kwargs) -> HybridRetriever:
        if provider not in cls.hybrid_retriver_providers:
            raise ValueError(
                f"Provider `{provider}` is not supported for Hybrid Retriever"
            )
        provider_class_name = (
            f"{_RETRIEVER_NAMESPACE_TRANSLATOR[provider]}HybridRetriever"
        )
        module_name = f"msgflow.data.retrivers.{provider}"
        hybrid_cls = import_module_from_lib(provider_class_name, module_name)
        retriever = hybrid_cls(**kwargs)
        return retriever