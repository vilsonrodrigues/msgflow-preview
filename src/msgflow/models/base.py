from abc import abstractmethod
from typing import Dict
from msgflow._private.core import BaseClient


class BaseModel(BaseClient):

    msgflow_type = "model"   
    to_remove = ["model", "processor", "client"]

    def instance_type(self) -> Dict[str, str]:
         return {"model_type": self.model_type}  

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
        }   

    @abstractmethod
    def _initialize_client(self):
        """
        Initialize the client. This method must be implemented by subclasses.

        This method is called during the deserialization process to ensure that the client
        is properly initialized after its state has been restored.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError