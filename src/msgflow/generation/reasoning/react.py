from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from msgspec import Struct


class FunctionCall(Struct, kw_only=True):
    id: Optional[UUID] = uuid4()
    name: str
    arguments: Optional[Dict[str, Any]]
    justification: Optional[str]
    result: Optional[str] = None


class Thought(Struct):
    reasoning: str
    plan: Optional[str] = None 


class ReActStep(Struct):
    thought: Thought
    actions: List[FunctionCall] = [] 


class ReActResult(Struct):
    answer: str 
    explanation: str 


class ReAct(Struct):
    current_step: Optional[ReActStep] = None
    final_answer: Optional[ReActResult] = None  
