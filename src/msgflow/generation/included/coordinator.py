from typing import List, Optional
from typing_extensions import Annotated
from msgspec import Meta, Struct
from msgflow.generation.templates import PromptSpec


EXAMPLES = Annotated[str, Meta(description=PromptSpec.EXAMPLES)]
EXPECTED_OUTPUT = Annotated[str, Meta(description=PromptSpec.EXPECTED_OUTPUT)]
INSTRUCTIONS = Annotated[str, Meta(description=PromptSpec.INSTRUCTIONS)]
SYSTEM_MESSAGE = Annotated[str, Meta(description=PromptSpec.SYSTEM_MESSAGE)]


class Analysis(Struct):
    problem: str
    plan: str


class Task(Struct):
    task: str
    member: str


class Member(Struct):
    name: str
    description: str
    system_message: SYSTEM_MESSAGE
    instructions: INSTRUCTIONS
    expected_output: Optional[EXPECTED_OUTPUT]
    examples: Optional[EXAMPLES]


class RemoveMember(Struct):
    name: str


class Coordinator(Struct):
    analysis: Optional[Analysis]
    tasks: Optional[List[Task]]
    final_answer: Optional[str]


class AutoCoordinator(Coordinator):
    new_members: Optional[List[Member]]
    remove_members: Optional[List[RemoveMember]]


COORDINATOR_SYSTEM_MESSAGE = """
You are a team coordinator. 

Your goal is to understand your task, identify the problems and draw up an action plan. 
You must distribute tasks to the team members.
Be precise when choosing tasks. You can assign more than one task to the same member.
You have a maximum number of iterations to define your final answer.
Write your responses in a structured way.

When you know it, you must write your final_answer.
"""

AUTO_COORDINATOR_SYSTEM_MESSAGE = COORDINATOR_SYSTEM_MESSAGE + """
You can add new expert members to solve the tasks and 
then assign them a task in the same iteration. 
You can also remove members from your team if you feel it is necessary.
"""