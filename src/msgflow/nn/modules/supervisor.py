from typing import List
from msgflow.nn.modules.agent import Agent
from msgflow.nn.modules.module import Module
from msgflow.nn import functional as F



class Supervisor(Module):

    def __init__(
        self, 
        head: Agent,
        workers: List[Module],
    ):
        super().__init__()
        self._set_head(head)
        self._set_head(head)

    def forward(self, message):
        F.bcast_gather()
    
    def _set_head(self, head: Agent):
        self.head = head