from __future__ import annotations
import inspect
from typing import Any, Callable, Iterator, List, Mapping, Sequence, Tuple
import gevent

from msgflow.nn.modules.container import ModuleDict
from msgflow.nn.modules.module import Module
from msgflow.utils import (
    convert_camel_to_snake_case,
    generate_json_schema,
)


# TODO: dynamic fns have dependencies, so they must be imported before running
# consider the possibility of having specific fn pools/libraries for each user
# tool call id tool_call_id = "vAHdf3"
# consider how to remove functions
# add temporary functions
# TODO: maximum number of functions
# TODO: provide the option of special functions that make the model have control over its own functions

class Tool(Module):
    """Tool class description"""

    def get_json_schema(self):
        return generate_json_schema(self)


def _convert_module_to_nn_tool(impl: Callable) -> Tool:
    """Convert a callable in nn.Tool"""
    if inspect.isclass(impl):
        if not hasattr(impl, "__call__"):
            raise NotImplementedError(
                "To transform a class in `nn.Tool`"
                " is necessary implement a `def __call__`"
            )

        if hasattr(impl, "__doc__") and impl.__doc__ is not None:
            doc = impl.__doc__
        elif hasattr(impl.__call__, "__doc__") and impl.__call__.__doc__ is not None:
            doc = impl.__call__.__doc__
        else:
            raise NotImplementedError(
                "To transform a class into a `nn.Tool` "
                "it is necessary to implement a docstring "
                "in the class or in `def __call__`"
            )

        if hasattr(impl, "__annotations__"):
            annotations = impl.__annotations__
        elif hasattr(impl.__call__, "__annotations__"):
            annotations = impl.__call__.__annotations__
        else:
            raise NotImplementedError(
                "To transform a class in `nn.Tool` is necessary "
                "to implement annotations of types hint in "
                "`self.__annotations__` or in `def __call__`"
            )

        name = convert_camel_to_snake_case(impl.__name__)

    elif inspect.isfunction(impl):
        if hasattr(impl, "__doc__") and impl.__doc__ is not None:
            doc = impl.__doc__
        else:
            raise NotImplementedError(
                "To transform a function into a `nn.Tool` "
                "is necessary to implement a docstring"
            )

        if hasattr(impl, "__annotations__"):
            annotations = impl.__annotations__
        else:
            raise NotImplementedError(
                "To transform a function into a `nn.Tool` "
                "is necessary to implement parameters "
                "annotations of types hint "
            )

        name = impl.__name__

    class WrappedTool(Tool):

        def __init__(self):
            super().__init__()
            self.set_name(name)
            self.set_description(doc)
            self._set_annotations(annotations)    
            super().__setattr__("impl", impl) # Not a buffer for now      

        def forward(self, *args, **kwargs):
            return self.impl(*args, **kwargs)

    return WrappedTool()


# implementar cancelamento de tarefas baseado em id
# usar o id da img
class ToolLibrary(Module):
    
    #_tasks = OrderedDict()  # TODO: para pensar aqui sobre cancelmanento de tasks

    def __init__(
        self,
        tools: List[Callable],
        # special_tools: Optional[List[str]] = None
    ):
        super().__init__()
        self.library = ModuleDict()
        for tool in tools:
            self.add(tool)

    def add(self, tool: Callable):
        if tool.__name__ in self.library.keys():
            raise ValueError(f"The tool name `{tool.name}` is already in tool library")
        if not isinstance(tool, Tool):
            tool = _convert_module_to_nn_tool(tool)
        self.library.update({tool.name: tool})

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            self.library.pop(tool_name)
        else:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")

    def clear(self):
        self.library.clear()

    def get_tools(self) -> Iterator[Mapping[str, Tool]]:
        return self.library.items()

    def get_functions_json_schema(self) -> Sequence[Mapping[str, Any]]:
        """ Returns a list of JSON schemas from functions """
        # TODO: para suportar fn que não sao necessariamente chamaveis via call
        # possa passar fns que nao sao Function, e ainda é necessario conseguir o json schema
        # entao use a fn original
        return [self.library[tool_name].get_json_schema() for tool_name in self.library]

    def forward(self, tool_callings: List[Tuple[str, str, Any]]) -> Mapping[str, str]:
        """ Execute tool calls.

        Args:
            tool_callings: 
                A list of tuples containing the tool id, name and parameters.
            
                !!! example

                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', '')]

        Returns:
            A list of dictionary containing the id of the tool
            and the result of the call.
            
            !!! example

                {'123121': '12:00', '322': '4 * 2 = 8'}
        """
        tool_results = {}
        greenlets = {}

        for id, name, args in tool_callings:
            if name in self.library.items():
                if args:
                    greenlet = gevent.spawn(self.library[name], **args)
                else:
                    greenlet = gevent.spawn(self.library[name])

                greenlets[id] = greenlet
            else:
                tool_results[id] = "This tool is not available"

        if greenlets:
            try:
                gevent.joinall(greenlets.values())

                # Collect results
                for id, greenlet in greenlets.items():
                    tool_results[id] = greenlet.value
            except gevent.Timeout:
                raise TimeoutError(f"Execution exceeded time limit")

        return tool_results