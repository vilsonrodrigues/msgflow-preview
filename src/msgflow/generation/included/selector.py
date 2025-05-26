from typing import Optional
from msgspec import Struct

class Selector(Struct):
    member: str


class SelectorReformulatedQuery(Selector):
    new_query: Optional[str]


SELECTOR_SYSTEM_MESSAGE = """
You are a team coordinator. 

Your goal is to analyze a task and choose one of the team members to solve it.
Write your responses in a structured way.
"""

SELECTOR_REFORMULATED_QUERY_SYSTEM_MESSAGE = SELECTOR_SYSTEM_MESSAGE + """
You can also rewrite the task the member will do instead of passing on the 
same one you receive, this can help make it clearer what the member should do.
"""
