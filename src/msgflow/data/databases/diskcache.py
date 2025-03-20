from typing import Any, Dict, List, Optional, Union
import msgspec
try:
    from diskcache import Cache
except:
    raise ImportError("`diskcache` is not detected, please install"
                      "using `pip install msgflow[diskcache]`")
from msgflow.data.databases.base import BaseDB
from msgflow.data.databases.types import KvDB
from msgflow.utils.convert import convert_str_to_hash


class DiskCacheKvDB(BaseDB, KvDB):

    provider: diskcache

    def __init__(self, ttl: Optional[int] = 3600, hash_key: Optional[bool] = True):
        self.hash_key = hash_key
        self.ttl = ttl
        self._initialize_client()

    def _initialize_client(self):
        self.client = Cache(timeout=1)
        
    def add(self, documents: Union[List[Dict[str, Any]], Dict[str, Any]]):
        if not isinstance(documents, list):
            documents = [documents]
        for document in documents:            
            for k, v in document.items():
                if self.hash_key:
                    k = convert_str_to_hash(k)
                v = msgspec.msgpack.encode(v)
                self.client.set(k, v, expire=self.ttl)
            
    def _search(self, queries: Union[str, List[str]]):
        if not isinstance(queries, list):
            queries = [queries]
        query_results = []
        for query in queries:
            if self.hash_key:
                query = convert_str_to_hash(query)
            v = self.client.get(query)
            if v is not None:
                v = msgspec.msgpack.decode(v)
            query_results.append(v)
        return query_results

    def __call__(self, queries: Union[str, List[str]]) -> Union[Any, List[Any]]:
        query_results = self._search(queries)
        if len(query_results) == 1:
            return query_results[0]
        return query_results
