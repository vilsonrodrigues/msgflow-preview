from typing import Dict, List
from msgspec import Struct
from typing_extensions import Generic, TypeVar


T = TypeVar("T", default=str)


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

SELF_CONSISTENCY_SYSTEM_MESSAGE = """

""".