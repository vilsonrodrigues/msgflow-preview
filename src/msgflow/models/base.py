from typing import Dict
from msgflow._private.core import BaseClient


class BaseModel(BaseClient):

    msgflow_type = "model"   
    to_remove = ["model", "processor", "client"]

    def instance_type(self) -> Dict[str, str]:
         return {"model_type": self.model_type}  