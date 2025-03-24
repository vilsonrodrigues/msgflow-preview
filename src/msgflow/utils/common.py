import inspect
from typing import Callable
from msgflow.nn.modules.module import Module


def get_callable_name(callable: Callable) -> str:
    if isinstance(callable, Module):
        return callable.get_module_name()
    elif inspect.isfunction(callable):    
        return callable.__name__
    else:
        return callable.__class__.__name__  
