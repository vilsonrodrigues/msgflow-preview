from typing import Dict, Generic, List, TypeVar
from msgspec import Struct

T = TypeVar("T")


class Solution(Struct):
    reasoning_steps: List[str]
    answer: str
    confidence_score: float


class SelfConsistency(Struct):
    solutions: List[Solution]
    most_common_answer: str
    confidence_distribution: Dict[str, float]
    final_answer: str
    explanation: str


class SelfConsistency(Struct, Generic[T]):
    solutions: List[Solution]
    most_common_answer: str
    confidence_distribution: Dict[str, float]
    final_answer: T
    explanation: str
