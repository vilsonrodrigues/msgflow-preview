from typing import Dict
from msgflow._private.core import BaseClient


class BaseParser(BaseClient):

    msgflow_type = "parser"
    to_remove = ["client"]

    def instance_type(self) -> Dict[str, str]:
         return {"parser_type": self.parser_type}  