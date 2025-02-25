from copy import deepcopy
from typing import Any, Optional
from typing_extensions import Self

class _CommonModuleData:

    def copy_to_data(self, data: Any):
        """ Copy new data to self.data """
        self.data = deepcopy(data)

    def clone(self) -> Self:
        return deepcopy(self)
    
    def copy_(self, src):
        """ Copies the elements from src into self tensor and returns self.

        The src tensor must be broadcastable with the self tensor. It may be of a different data type or reside on a different device.

        Parameters
        src (Tensor): the source tensor to copy from

        non_blocking (bool)  if True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect."""
        if src is not None:
            self.data = deepcopy(src)
        

class Parameter(_CommonModuleData):

    """
    Parameter is a prompt component in `nn.Module` that can be optimized.

    Parameters that have a very special property when used with `Module`s 
    - when they're assigned as Module attributes they are automatically 
    added to the list of its parameters, and will appear e.g. in 
    `Module.parameters` iterator.

    Args:
        data: Prompt component content
        spec: Prompt component specification
        requires_grad: If the parameter requires "gradient"
    """

    grad: Optional[str] = None

    def __init__(self, data: str, spec: str, requires_grad: Optional[bool] = True):
        self.data = data
        self.spec = spec
        self.requires_grad = requires_grad
    
    def __hash__(self):
        return hash((self.data, self.spec))
    
    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return False
        return (self.data == other.data and self.spec == other.spec)
    
    def __repr__(self):
        return f"Parameter(data='{self.data}', spec='{self.spec}', requires_grad={self.requires_grad})"

    def requires_grad_(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class Buffer(_CommonModuleData):
    """
    Buffer is a kind of `Parameter` that cannot be optimized.
    
    Buffers should be used to store information about the 
    operation of a `Module`. This allows the generation of a 
    checkpoint of a Module. Buffers can be any type of python object.
    
    Buffers has a very special property when used with `Module`s 
    when they're assigned as Module attributes they are automatically 
    added to the list of its buffers, and will appear e.g. in buffers() 
    iterator.

    Args:
        data: Python object
        persistent: Whether the buffer is part of the module's state_dict. Default: True
    """

    data: Any

    def __init__(self, data: Any, *, persistent: Optional[bool] = True):
        self.data = data
        self.persistent = persistent

    def __hash__(self):
        return hash(self.data)
    
    def __eq__(self, other):
        if not isinstance(other, Buffer):
            return False
        return self.data == other.data

    def __repr__(self):
        return "Buffer containing:\n" + super().__repr__()