from typing import Dict, Generic, List, Optional, Union, TypeVar
from enum import Enum
from msgspec import Struct

T = TypeVar("T")


class StepType(Enum):
    OBSERVATION = "observation"
    CALCULATION = "calculation"
    DEDUCTION = "deduction"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"


class Difficulty(Enum):
    BASIC = "basic"
    MIDDLE = "middle"
    HIGH = "high"


class Context(Struct):
    domain: str
    difficulty: Difficulty
    required_knowledge: str


class Step(Struct):
    step_type: StepType
    explanation: str
    output: str
    confidence: float
    intermediate_results: Optional[Dict[str, Union[str, float, int]]]
    steps_dependencies_idx: Optional[List[int]] = []


class ValidationStep(Struct):
    method: str
    result: bool
    error_margin: Optional[float]
    explanation: str


class ChainOfThoughts(Struct, Generic[T]):
    context: Optional[Context]
    assumptions: List[str] = []
    steps: List[Step]
    final_answer: T
    validation: ValidationStep
    confidence_score: float
    alternative_approaches: Optional[List[str]] = []
