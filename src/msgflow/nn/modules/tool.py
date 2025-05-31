import inspect
from typing import Any, Callable, Dict, Iterator, List, Tuple

from msgflow.nn.modules.container import ModuleDict
from msgflow.nn import functional as F
from msgflow.nn.modules.module import Module
from msgflow.utils.chat import generate_tool_json_schema
from msgflow.utils.convert import convert_camel_to_snake_case
from msgflow.utils.tenacity import tool_retry
from msgflow.telemetry.span import trace_tool_library_call

# TODO: dynamic fns have dependencies, so they must be imported before running
# consider the possibility of having specific fn pools/libraries for each user
# tool call id tool_call_id = "vAHdf3"
# consider how to remove functions
# add temporary functions
# TODO: maximum number of functions
# TODO: provide the option of special functions that make the model have control over its own functions
# TODO: mcp

class ToolBase(Module):
    """Tool class description"""

    def get_json_schema(self):
        return generate_tool_json_schema(self)


def _convert_module_to_nn_tool(impl: Callable) -> ToolBase:
    """Convert a callable in nn.Tool"""
    # Case 1: Uninitialized or initialized class
    if inspect.isclass(impl) or callable(impl):
        if not hasattr(impl, "__call__"):
            raise NotImplementedError(
                "To transform a class in `nn.Tool`"
                " is necessary implement a `def __call__`"
            )

        if hasattr(impl, "docstring") and impl.docstring is not None:
            doc = impl.docstring
        elif hasattr(impl, "__doc__") and impl.__doc__ is not None:
            doc = impl.__doc__
        elif hasattr(impl.__call__, "__doc__") and impl.__call__.__doc__ is not None:
            doc = impl.__call__.__doc__
        else:
            raise NotImplementedError(
                "To transform a class into a `nn.Tool` "
                "it is necessary to implement a docstring. "
                "Can be: a cls attr `self.docstring`, or"
                "a docstring in the class or in `def __call__`"
            )

        if hasattr(impl, "annotations"):
            annotations = impl.annotations
        elif hasattr(impl, "__annotations__"):
            annotations = impl.__annotations__
        elif hasattr(impl.__call__, "__annotations__"):
            annotations = impl.__call__.__annotations__
        else:
            raise NotImplementedError(
                "To transform a class in `nn.Tool` is necessary "
                "to implement annotations of types hint in "
                "`self.annotations`, `self.__annotations__` or in `def __call__`"
            )

        name = convert_camel_to_snake_case(impl.__name__)

        if inspect.isclass(impl):
            impl = impl() # Initialized

    # Case 2: Function
    elif inspect.isfunction(impl) or inspect.iscoroutinefunction(impl):
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
        
    else:
        raise ValueError("The given object is not a callable function, class, or instance")

    class Tool(ToolBase):

        def __init__(self):
            super().__init__()
            self.set_name(name)
            self.set_description(doc)
            self._set_annotations(annotations)    
            self.impl = impl # Not a buffer for now      

        @tool_retry
        def forward(self, *args, **kwargs):
            if inspect.iscoroutinefunction(self.impl):
                return F.wait_for(self.impl, *args, **kwargs)
            return self.impl(*args, **kwargs)

    return Tool()


# implementar cancelamento de tarefas baseado em id
# usar o id da img
class ToolLibrary(Module):
    
    def __init__(
        self,
        name: str,
        tools: List[Callable],
        # special_tools: Optional[List[str]] = None
    ):
        super().__init__()
        self.set_name(f"{name}_tool_library")
        self.library = ModuleDict()
        for tool in tools:
            self.add(tool)

    def add(self, tool: Callable):
        if tool.__name__ in self.library.keys():
            raise ValueError(f"The tool name `{tool.__name__}` is already in tool library")
        if not isinstance(tool, ToolBase):
            tool = _convert_module_to_nn_tool(tool)
        self.library.update({tool.name: tool})

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            self.library.pop(tool_name)
        else:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")

    def clear(self):
        self.library.clear()

    def get_tools(self) -> Iterator[Dict[str, ToolBase]]:
        return self.library.items()

    def get_tool_names(self) -> List[str]:
        return list(self.library.keys())

    def get_tool_json_schemas(self) -> List[Dict[str, Any]]:
        """ Returns a list of JSON schemas from functions """
        # TODO: para suportar fn que não sao necessariamente chamaveis via call
        # possa passar fns que nao sao Function, e ainda é necessario conseguir o json schema
        # entao use a fn original
        return [self.library[tool_name].get_json_schema() for tool_name in self.library]

    @trace_tool_library_call
    def forward(self, tool_callings: List[Tuple[str, str, Any]]) -> Dict[str, str]:
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
        tool_responses = {}

        tool_names = self.get_tool_names()

        messages = []
        to_send = []
        tool_ids = []

        for id, name, args in tool_callings:
            elif name in tool_names:
                ...
            elif name in tool_names:
                if args:
                    messages.append(**args)
                else:
                    messages.append(None)
                to_send.append(self.library[name])
                tool_ids.append(id)
            else:
                tool_responses[id] = "This tool is not available"

        if messages and to_send:
            responses = F.scatter_gather(to_send, kwargs_list=messages)
            for id, response in zip(tool_ids, responses):
                tool_responses[id] = response

        return tool_responses