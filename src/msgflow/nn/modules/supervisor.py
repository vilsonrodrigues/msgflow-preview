from typing import List
from msgflow.nn.modules.agent import Agent
from msgflow.nn.modules.module import Module
from msgflow.nn import functional as F
from msgflow.utils.chat import format_available_modules


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


        F.scatter_gather(messages, [])
    
    def _set_head(self, head: Agent):
        if isinstance(head, Agent):
            self.head = head
            #format_available_modules()
            self.head._set_system_extra_message()
        else:
            raise ValueError("")

    def _set_workers(self, workers: List[Module]):
        if all(isinstance(worker, Module) for worker in workers):
            self.workers = workers
        else:
            raise ValueError("")