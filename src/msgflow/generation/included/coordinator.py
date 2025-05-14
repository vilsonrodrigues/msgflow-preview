from typing import List, Optional
from msgspec import Struct


class Analyze(Struct):
    domain: str
    problem: str
    plan: str

class SubTask(Struct):
    task: str
    worker: str

class Coordinator(Struct):
    analyze: Optional[Analyze]
    sub_tasks = Optional[List[SubTask]]
    final_answer: Optional[str]

AVAILABLE_WORKERS_TEMPLATE =  """

{}
"""