from typing import Dict
from msgflow._private.core import BaseClient


class BaseRetriever(BaseClient):

    msgflow_type = "retriever"
    to_remove = ["client"]

    def instance_type(self) -> Dict[str, str]:
        return {"parser_type": self.retriever_type}  