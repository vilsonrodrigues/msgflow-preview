from typing import Dict, Generic, List, Optional, TypeVar
from enum import Enum
from msgspec import Struct

T = TypeVar("T")


class ThoughtState(Enum):
    PROMISING = "promising"
    NEUTRAL = "neutral"
    DEAD_END = "dead_end"


class Thought(Struct):
    content: str
    evaluation: ThoughtState
    score: float
    reasoning: str


class ThoughtNode(Struct):
    thought: Thought
    children: Optional[List["ThoughtNode"]] = []
    depth: int
    branch_id: str


class TreeExploration(Struct):
    current_node: ThoughtNode
    promising_paths: List[List[Thought]]
    dead_ends: List[List[Thought]]
    evaluation_metrics: Dict[str, float]


class TreeOfThoughts(Struct, Generic[T]):
    initial_thoughts: List[Thought]
    exploration_tree: ThoughtNode
    best_path: List[Thought]
    confidence_score: float
    reasoning_summary: str
    final_answer: T
