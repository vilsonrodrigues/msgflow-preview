from msgflow.utils.imports import import_module_from_lib
from msgflow.data.databases.types import (
    GeoDB,
    GraphDB,
    InMemoryDB,
    NoSQLDB,
    RelationalDB,
    TimeSeriesDB,
    VectorDB,
)

_SUPPORTED_DB_TYPES = [
    "vector",
    "graph",
    "relational",
    "nosql",
    "in_memory",
    "time_series",
    "geo",
]
_DB_NAMESPACE_TRANSLATOR = {
    "qdrant": "Qdrant",
    "postgres": "Postgres",
    "weavite": "Weavite",
    "milvus": "Milvus",
    "lance_db": "LanceDB",
    "vespa": "Vespa",
    "pinecone": "Pinecone",
    "cassandra": "Cassandra",
    "redis": "Redis",
    "deep_lake": "DeepLake",
    "marqo": "Marqo",
    "faiss": "FAISS",
    "azure": "AzureAISearch",
    "supabase": "SupaBase",
    "elastic_search": "ElasticSearch",
    "chroma": "Chroma",
    "sqlite": "SQLite",
    "neo4j": "Neo4J",
}
_VECTOR_DB_PROVIDERS = [
    "qdrant",
    "postgres",
    "weavite",
    "milvus",
    "lance_db",
    "vespa",
    "redis",
    "cassandra",
    "deep_lake",
    "marqo",
    "elastic_search",
    "sqlite",
    "faiss",
    "chroma",
]
_GRAPH_DB_PROVIDERS = ["neo4j"]


class DataBase:
    supported_db_types = _SUPPORTED_DB_TYPES
    vector_db_providers = _VECTOR_DB_PROVIDERS
    graph_db_providers = _GRAPH_DB_PROVIDERS

    @classmethod
    def vector(cls, provider: str, **kwargs) -> VectorDB:
        if provider not in cls.vector_db_providers:
            raise ValueError(f"Provider `{provider}` is not supported for Vector DB.")
        provider_class_name = f"{_DB_NAMESPACE_TRANSLATOR[provider]}VectorDB"
        module_name = f"msgflow.data.databases.{provider}"
        vector_db_cls = import_module_from_lib(provider_class_name, module_name)
        db = vector_db_cls(**kwargs)
        return db

    @classmethod
    def graph(cls, provider: str, **kwargs) -> GraphDB:
        if provider not in cls.graph_db_providers:
            raise ValueError(f"Provider `{provider}` is not supported for Graph DB.")
        provider_class_name = f"{_DB_NAMESPACE_TRANSLATOR[provider]}GraphDB"
        module_name = f"msgflow.data.databases.{provider}"
        graph_db_cls = import_module_from_lib(provider_class_name, module_name)
        db = graph_db_cls(**kwargs)
        return db