from abc import abstractmethod
from typing import Dict
from msgflow._private.core import BaseClient


class BaseDB(BaseClient):

    msgflow_type = "db"
    to_remove = ["client"]

    def instance_type(self) -> Dict[str, str]:
         return {"db_type": self.db_type}  
    
    @abstractmethod
    def add(self):
        """
        Insert data on database
        """
        raise NotImplementedError    