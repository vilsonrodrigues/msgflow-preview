from functools import wraps
from types import FunctionType, MethodType
from typing import Callable, Dict, Optional, Union
from msgflow.dotdict import dotdict


def tool_props(
    *, 
    return_direct: Optional[bool] = False, 
    name_override: Optional[str] = None, 
    handoff: Optional[bool] = False
) -> Callable:
    """Decorator to inject custom properties into a function or class instance.

    This decorator adds metadata properties to a function or an instance of a class, allowing 
    tools to control behavior such as whether results are returned directly or passed for 
    further handling, and optionally override the tool's registered name.

    Args:
        return_direct: 
            If True, the tool will return its output directly without additional processing. 
        name_override: 
            A custom name to override the default tool name derived from the function 
            or class. If not provided, the original name is used.
        handoff: 
            If True, indicates that this function will receive the `model_state` from the Agent.
    Returns:
        A decorator that modifies the target function or class instance 
        by injecting the specified properties.
    """
    def decorator(f):
        props = {
            "props": dotdict({
                "return_direct": return_direct,
                "handoff": handoff,
            })
        }
        if isinstance(f, (FunctionType, MethodType)):
            return decorate_function(f, name_override, props)
        if isinstance(f, type): # Not initialized class
            f = f() # Init class
        return decorate_instance(f, name_override, props)
    return decorator

def decorate_function(
    func: Union[FunctionType, MethodType], 
    override_name: str, 
    props: Dict[str, Union[bool, str]]
) -> Union[FunctionType, MethodType]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    if override_name:
        wrapper.__name__ = override_name
        wrapper.__qualname__ = override_name

    wrapper.__dict__.update(props)
    return wrapper

def decorate_instance(
    instance: Callable, 
    override_name: str, 
    props: Dict[str, Union[bool, str]]
) -> Callable:
    if override_name:
        instance.__name__ = override_name
        instance.__qualname__ = override_name

    instance.__dict__.update(props)
    return instance
