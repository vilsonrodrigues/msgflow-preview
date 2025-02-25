import functools
import inspect
import weakref
from collections import namedtuple, OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import msgspec
from code2mermaid import code_to_mermaid
from jinja2 import Template  

import msgflow
from msgflow._internal.core import Core
from msgflow.message_otel import Message
from msgflow.models.model import Model
from msgflow.nn.parameter import Buffer, Parameter
from msgflow.utils_._msgspec import (
    deserialize_struct, 
    serialize_msgspec_struct
)
from msgflow.utils_.hooks import RemovableHandle
from msgflow.utils_.plot import plot_mermaid
from msgflow.utils_.validation import is_builtin_type


__all__ = [
    "register_module_forward_pre_hook",
    "register_module_forward_hook",
    "register_module_buffer_registration_hook",
    "register_module_module_registration_hook",
    "register_module_parameter_registration_hook",
    "Module",
]

MSGFLOW_DESERIALIZABLE_CLS: Dict[str, Type] = {
    "model": Model
}


T = TypeVar("T", bound="Module")

# TODO para serializar basta verificar se o obj tem msgflow_type
# isso resolve em vez de ficar add manualmente quais são

class _IncompatibleKeys( # TODO tirar. tirar nao porra. tem que ficar
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__

def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: Dict[int, Callable] = OrderedDict()

class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["Module"] = None):
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> Dict:
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            result["module"] = self.module()

        return result

    def __setstate__(self, state: Dict):
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError(
                    "You are trying to revive the hook of a dead Module!"
                )
            self.module = weakref.ref(state["module"])


r"""This tracks hooks common to all modules that are executed before/after
calling forward. This is global state used for debugging/profiling
purposes"""
_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: Dict[int, bool] = OrderedDict()
_global_forward_hooks_with_kwargs: Dict[int, bool] = OrderedDict()

_EXTRA_STATE_KEY_SUFFIX = "_extra_state"

def register_module_buffer_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a buffer registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_buffer` is invoked.
    It should have the following signature::

        hook(module, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`msgflow.nn.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a module registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`msgflow.nn.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a parameter registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`msgflow.nn.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`msgflow.nn.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(
    hook: Callable[..., None],
    *,
    with_kwargs: bool = False,
    always_call: bool = False,
) -> RemovableHandle:
    r"""Register a global forward hook for all the modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Parameters:
        hook (Callable): The user defined hook to be registered.
        always_call (bool): If ``True`` the ``hook`` will be run regardless of
            whether an exception is raised while calling the Module.
            Default: ``False``
    Returns:
        :class:`msgflow.nn.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = RemovableHandle(
        _global_forward_hooks, extra_dict=_global_forward_hooks_always_called
    )
    _global_forward_hooks[handle.id] = hook
    if with_kwargs:
        _global_forward_hooks_with_kwargs[handle.id] = True
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    return handle


def _forward_unimplemented(self, *input: Any) -> None:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(
        f"Module [{type(self).__name__}] is missing the required `forward` function"
    )

class Module(Core):

    training: bool # TODO mover para Agent

    _version: int = 1
    r"""This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""

    _parameters: Dict[str, Optional[Parameter]] = OrderedDict()
    _buffers: Dict[str, Optional[Buffer]] = OrderedDict()
    _modules: Dict[str, Optional["Module"]] = OrderedDict()
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_hooks_with_kwargs: Dict[int, bool]
    # forward hooks that should always be called even if an exception is raised
    _forward_hooks_always_called: Dict[int, bool]
    _forward_pre_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_pre_hooks_with_kwargs: Dict[int, bool]
    _state_dict_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _state_dict_pre_hooks: Dict[int, Callable]
    _load_state_dict_post_hooks: Dict[int, Callable]
    call_super_init: bool = False    

    def __init__(self, *args, **kwargs) -> None:
        """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
        # torch._C._log_api_usage_once("python.nn_module")

        # Backward compatibility: no args used to be allowed when call_super_init=False
        if self.call_super_init is False and bool(kwargs):
            raise TypeError(
                f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
                ""
            )

        if self.call_super_init is False and bool(args):
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were"
                " given"
            )

        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__("training", True) # mover para agent
        super().__setattr__("_parameters", {})
        super().__setattr__("_buffers", {})
        super().__setattr__("_non_persistent_buffers_set", set()) # ?
        super().__setattr__("_forward_hooks", OrderedDict())
        super().__setattr__("_forward_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_forward_hooks_always_called", OrderedDict())
        super().__setattr__("_forward_pre_hooks", OrderedDict())
        super().__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_state_dict_hooks", OrderedDict())
        super().__setattr__("_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_post_hooks", OrderedDict())
        super().__setattr__("_modules", {})

        if self.call_super_init:
            super().__init__(*args, **kwargs)

    forward: Callable[..., Any] = _forward_unimplemented

    # msgflow funcs

    def _get_mermaid(
        self, title: Optional[str] = None, orientation: Optional[str] = "TD"
    ) -> str:
        mermaid = code_to_mermaid(
            inspect.getsource(self.forward),
            remove_self=True,
            title=title,
            orientation=orientation,
        )
        return mermaid

    def plot(self, title: Optional[str] = None, orientation: Optional[str] = "TD"):
        mermaid = self.get_mermaid(title, orientation)
        return plot_mermaid(mermaid)

    def _get_content_from_or_input(self, path: str, message: Message) -> Any:
        """Returns the first valid content from OR input"""
        content = None
        for single_path in path:
            content = message.get(single_path)
            if content is not None:
                break
        return content

    def _format_task_template(self, content: Union[str, Dict[str, Any]]) -> str:
        return self._format_template(content, self.task_template)

    def _format_response_template(self, content: str) -> str:
        return self._format_template(content, self.response_template)    

    def _format_template(self, content: Union[str, Dict[str, Any]], raw_template: str) -> str:
        if isinstance(content, str):
            # Convert jinja to string format template
            template = raw_template.replace("{{ }}", "{}").replace("{{}}", "{}")
            return template.format(content)
        elif isinstance(raw_template, dict):
            template = Template(self.task_template)
            return template.render(raw_template)
        else:
            raise ValueError("Unsupported content type for template formatting")    

    def set_name(self, name: str):
        if isinstance(name, str):
            if name != "":
                self.register_buffer("name", name)            
            else:
                raise ValueError("`name` requires a string not empty")
        else:
            raise TypeError(f"`name` need be a `str` given {type(name)}")
            
    def _set_annotations(self, annotations: Dict[str, type]):
        if isinstance(annotations, dict):
            self.__annotations__ = annotations
        else:
            raise TypeError(f"`annotations` need be a `dict` given {type(annotations)}")
                        
    # TODO ambos os casos eu devo poder por em tool library pra acessar esse buffer
    # por meio de um get_annotation ou get description
    def set_description(self, description: str):
        if isinstance(description, str):
            self.register_buffer("__doc__", description)
        else:
            raise ValueError("`description` requires a string not empty")

    # msgflow END

    def register_buffer(self, name: str, data: Any, persistent: bool = True) -> None:
        # TODO: muito trabalho pra ajeitar a docstring
        # mudei de tensor para data
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {type(name)}")
        elif "." in name:
            raise KeyError("buffer name can't contain '.'")
        elif name == "":
            raise KeyError("buffer name can't be empty string")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif data is None:
            raise KeyError("buffer data can't be None")
        else:
            if isinstance(data, Buffer):
                buffer = data
                buffer.persistent = persistent
            else:
                buffer = Buffer(data=data, persistent=persistent)
            
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, buffer)
                if output is not None:
                    buffer = output
            
            self._buffers[name] = buffer
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)
                
    def register_parameter(self, name: str, param: Parameter) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(name, str):
            raise TypeError(
                f"parameter name should be a string. Got {type(name)}"
            )
        elif "." in name:
            raise KeyError("parameter name can't contain '.'")
        elif name == "":
            raise KeyError("parameter name can't be empty string")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        elif param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{type(param)}' object to parameter '{name}' "
                "(msgflow.nn.Parameter required)"
            )
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def add_module(self, name: str, module: "Module") -> None:
        r"""Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {type(name)}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f"module name can't contain '.', got: {name}")
        elif name == "":
            raise KeyError("module name can't be empty string ")
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        self._modules[name] = module

    def register_module(self, name: str, module: "Module") -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        #mod: msgflow.nn.Module = self
        mod = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no " "attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            #if not isinstance(mod, msgflow.nn.Module):
            #    raise AttributeError("`" + item + "` is not " "an nn.Module")

        return mod

    def set_submodule(self, target: str, module: "Module") -> None:
        """
        Set the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To overide the ``Conv2d`` with a new submodule ``Linear``, you
        would call
        ``set_submodule("net_b.net_c.conv", nn.Linear(33, 16))``.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)
            module: The module to set the submodule to.

        Raises:
            ValueError: If the target string is empty
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        atoms: List[str] = target.split(".")
        name = atoms.pop(-1)
        mod: msgflow.nn.Module = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            # Use isinstance instead of type here to also handle subclass of nn.Module
            if not isinstance(mod, msgflow.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        setattr(mod, name, module)

    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")

        #mod: msgflow.nn.Module = self.get_submodule(module_path)
        mod = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        #param: msgflow.nn.Parameter = getattr(mod, param_name)
        param = getattr(mod, param_name)

        #if not isinstance(param, msgflow.nn.Parameter):
        #    raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

        return param

    def get_buffer(self, target: str) -> "Buffer":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")

        #mod: msgflow.nn.Module = self.get_submodule(module_path)
        mod = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        #buffer: msgflow.Buffer = getattr(mod, buffer_name)
        buffer = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    # revisar

    def register_forward_pre_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...]], Optional[Any]],
            Callable[
                [T, Tuple[Any, ...], Dict[str, Any]],
                Optional[Tuple[Any, Dict[str, Any]]],
            ],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.


        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`msgflow.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`msgflow.nn.modules.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``
            with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
                given to the forward function.
                Default: ``False``

        Returns:
            :class:`msgflow.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(
            self._forward_pre_hooks, extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.

        If ``with_kwargs`` is ``False`` or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        output. It can modify the input inplace but it will not have effect on
        forward since this is called after :func:`forward` is called. The hook
        should have the following signature::

            hook(module, args, output) -> None or modified output

        If ``with_kwargs`` is ``True``, the forward hook will be passed the
        ``kwargs`` given to the forward function and be expected to return the
        output possibly modified. The hook should have the following signature::

            hook(module, args, kwargs, output) -> None or modified output

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`msgflow.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`msgflow.nn.modules.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
            with_kwargs (bool): If ``True``, the ``hook`` will be passed the
                kwargs given to the forward function.
                Default: ``False``
            always_call (bool): If ``True`` the ``hook`` will be run regardless of
                whether an exception is raised while calling the Module.
                Default: ``False``

        Returns:
            :class:`msgflow.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(
            self._forward_hooks,
            extra_dict=[
                self._forward_hooks_with_kwargs,
                self._forward_hooks_always_called,
            ],
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)
        return handle

    def _call_impl(self, *args, **kwargs):
        # Se não houver hooks, podemos simplificar o fluxo
        if not (self._forward_hooks or self._forward_pre_hooks):
            return self._call(*args, **kwargs)
        
        # Processa forward pre hooks
        for hook in self._forward_pre_hooks.values():
            if hook.__kwdefaults__ and "kwargs" in hook.__kwdefaults__:
                hook_result = hook(self, args, kwargs)
                if hook_result is not None:
                    if isinstance(hook_result, tuple) and len(hook_result) == 2:
                        args, kwargs = hook_result
                    else:
                        raise RuntimeError("forward pre-hook must return None or "
                                           "a tuple of (new_args, new_kwargs)")
            else:
                hook_result = hook(self, args)
                if hook_result is not None:
                    if not isinstance(hook_result, tuple):
                        hook_result = (hook_result,)
                    args = hook_result
        
        # Executa o forward com verificação de Message
        result = self._call(*args, **kwargs)
        
        # Processa forward hooks
        for hook in self._forward_hooks.values():
            if hook.__kwdefaults__ and "kwargs" in hook.__kwdefaults__:
                hook_result = hook(self, args, kwargs, result)
            else:
                hook_result = hook(self, args, result)
                
            if hook_result is not None:
                result = hook_result
                
        return result

    def _call(self, *args, **kwargs):
        # Search for a Message in args and kwargs
        message = next(
            (arg for arg in args if isinstance(arg, Message)),
            next(
                (v for v in kwargs.values() if isinstance(v, Message)),
                None
            )
        )
        
        # TODO: isso aqui deverá ter uma env que le em runtime
        # e por default ela é false
        if message is not None:
            # Check if this module has already processed the message
            # If it is True, skip the module
            if self.in_msg(self.name):
                return message
        
        module_output = self.forward(*args, **kwargs)
        return module_output

    __call__: Callable[..., Any] = _call_impl

    #def __getstate__(self): TODO deprecated. agora herda de Core
    #    state = self.__dict__.copy()
    #    return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Support loading old checkpoints that don't have the following attrs:
        if "_forward_pre_hooks" not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if "_forward_pre_hooks_with_kwargs" not in self.__dict__:
            self._forward_pre_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_with_kwargs" not in self.__dict__:
            self._forward_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_always_called" not in self.__dict__:
            self._forward_hooks_always_called = OrderedDict()
        if "_state_dict_hooks" not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if "_state_dict_pre_hooks" not in self.__dict__:
            self._state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_pre_hooks" not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_post_hooks" not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()
        if "_non_persistent_buffers_set" not in self.__dict__:
            self._non_persistent_buffers_set = set()

    # It is crucial that the return type is not annotated as `Any`, otherwise type checking
    # on `torch.nn.Module` and all its subclasses is largely disabled as a result. See:
    # https://github.com/pytorch/pytorch/pull/115074
    def __getattr__(self, name: str) -> Union[Any, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Union[Any, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{type(value)}' as parameter '{name}' "
                    "(msgflow.nn.Parameter or None expected)"
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign '{type(value)}' as child module '{name}' "
                        "(torch.nn.Module or None expected)"
                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if isinstance(value, Buffer) or buffers is not None and name in buffers:
                    #if value is not None and not isinstance(value, torch.Tensor):
                    #    raise TypeError(
                    #        f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                    #        "(torch.nn.Buffer, torch.Tensor or None expected)"
                    #    )
                    if isinstance(value, Buffer):
                        persistent = value.persistent
                    else:
                        persistent = name not in self._non_persistent_buffers_set
                    # === HACK ===
                    # This whole block below should just be:
                    # self.register_buffer(name, value, persistent)

                    # But to support subclasses of nn.Module that (wrongfully) implement a
                    # register_buffer() method that doesn't have the "persistent"
                    # argument. Only pass it in if it is accepted otherwise assume
                    # it is always true
                    if self.register_buffer is msgflow.nn.Module.register_buffer:
                        self.register_buffer(name, value, persistent)
                    else:
                        sign = inspect.signature(self.register_buffer)
                        if "persistent" in sign.parameters:
                            self.register_buffer(name, value, persistent)
                        else:
                            if not persistent:
                                raise RuntimeError(
                                    "Registering a non-persistent buffer "
                                    "on a Module subclass that implements "
                                    "register_buffer() without the persistent "
                                    "argument is not allowed."
                                )
                            # Assume that the implementation without the argument has the
                            # behavior from before the argument was added: persistent=True
                            self.register_buffer(name, value)
                    # === HACK END ===
                else:
                    super().__setattr__(name, value)

    # NOVA ANALISE

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _register_state_dict_hook(self, hook):
        r"""Register a post-hook for the :meth:`~msgflow.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None or state_dict

        The registered hooks can modify the ``state_dict`` inplace or return a new one.
        If a new ``state_dict`` is returned, it will only be respected if it is the root
        module that :meth:`~nn.Module.state_dict` is called from.
        """
        if getattr(hook, "_from_public_api", False):
            raise RuntimeError(
                "Cannot register the same function as the state dict post hook that was "
                "previously registered via register_state_dict_post_hook"
            )
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_post_hook(self, hook):
        r"""Register a post-hook for the :meth:`~msgflow.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None

        The registered hooks can modify the ``state_dict`` inplace.
        """
        # In _register_state_dict_hook there was a bug described in
        # https://github.com/pytorch/pytorch/issues/117437 where the return value
        # was only respected for the root module but not child submodules.
        # We fix this in this public version by only allowing inplace modifications on
        # the state_dict by the hook. However, since hooks registered via both these
        # APIs will be added to `_state_dict_hooks` and the type of `_state_dict_hooks`
        # cannot be changed due to many dependencies on it, we mark a hook
        # as being registered via the public API by setting `_from_public_api` on it.
        # In the implementation of `state_dict`, if the callable does not have this
        # flag, the old behavior of respecting the return value will be preserved
        # for the root module, otherwise, we ensure that the hook returns None.
        hook._from_public_api = True
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook for the :meth:`~msgflow.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, prefix, keep_vars) -> None

        The registered hooks can be used to perform pre-processing before the ``state_dict``
        call is made.
        """
        handle = RemovableHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle

    def _get_serializable_value(self, obj):
        """ Get serializable value from an object """
        if is_builtin_type(obj):
            return obj
        elif issubclass(obj, msgspec.Struct):
            return serialize_msgspec_struct(obj)
        elif hasattr(obj, "serialize"):
            return obj.serialize()
        else:
            # Fallback: convert to string if no other option
            return str(obj)

    def _save_to_state_dict(self, destination, prefix):
        """ Save parameters and buffers to state dict """
        # Save parameters (only the data string)
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data

        # Save buffers (handle different data types)
        for name, buf in self._buffers.items():
            if buf is not None and buf.persistent:
                destination[prefix + name] = self._get_serializable_value(buf.data)

    def state_dict(
        self, 
        destination: Optional[Dict[str, Any]] = None, 
        prefix: Optional[str] = ""
    ):
        """
        Returns a dictionary containing module's state.
        
        Args:
            destination: 
                If provided, the state will be updated into
                the given dict. Default: None
            prefix: 
                Prefix added to parameter and buffer names.
                Default: ""
        """
        if destination is None:
            destination = {}

        # Save current module's state
        self._save_to_state_dict(destination, prefix)

        # Save states from child modules
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + '.'
                )

        return destination

    def _register_load_state_dict_pre_hook(self, hook, with_module=False):
        r"""See :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` for details.

        A subtle difference is that if ``with_module`` is set to ``False``, then the
        hook will not take the ``module`` as the first argument whereas
        :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` always takes the
        ``module`` as the first argument.

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
        handle = RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = _WrappedHook(
            hook, self if with_module else None
        )
        return handle

    def register_load_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
        """
        return self._register_load_state_dict_pre_hook(hook, with_module=True)

    def register_load_state_dict_post_hook(self, hook):
        r"""Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.id] = hook
        return handle

    def _load_from_state_dict(self, state_dict: Dict[str, Any], prefix: Optional[str] = "") -> None:
        """
        Loads the module state from a state dict.

        Args:
            state_dict: Dictionary containing the state
            prefix: Prefix used for parameter/buffer names
        """
        # Load parameters
        for name, param in self._parameters.items():
            if param is not None:
                key = prefix + name
                if key in state_dict:
                    self._parameters[name].copy_to_data(state_dict[key])                    

        # Load buffers
        for name, buf in self._buffers.items():
            if buf is not None and buf.persistent:
                key = prefix + name
                if key in state_dict:
                    data = state_dict[key]
                    # Check if it is a msgflow serializable class
                    if isinstance(data, dict) and "msgflow_type" in data:
                        msgflow_type = data.pop("msgflow_type")
                        if msgflow_type in MSGFLOW_DESERIALIZABLE_CLS:
                            cls = MSGFLOW_DESERIALIZABLE_CLS[msgflow_type]
                            instance = cls.from_serialized(**data)
                            self._buffers[name].copy_to_data(instance)
                        elif msgflow_type == "generation_schema":
                            state = data.pop("state")
                            generation_schema = deserialize_struct(state)
                            self._buffers[name].copy_to_data(generation_schema)
                    else:
                        # Otherwise, load the value directly
                        self._buffers[name].data = data

        # Load submodules recursively
        for name, module in self._modules.items():
            if module is not None:
                module_prefix = prefix + name + "."
                module_dict = {
                    k.replace(module_prefix, ""): v 
                    for k, v in state_dict.items() 
                    if k.startswith(module_prefix)
                }
                if module_dict:
                    module._load_from_state_dict(module_dict)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the state of the module and its submodules.

        Args:
            state_dict: Dictionary containing the complete state
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"`state_dict` to be dict, given {type(state_dict).__name__}")
            
        self._load_from_state_dict(state_dict)

    def _named_members( # ok?
        self, get_members_fn, prefix="", recurse=True, remove_duplicate=True
    ):
        r"""Help yield various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse: If True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Returns:
            Parameter: module parameter

        !!! example 
            # TODO
            ```python
            for param in model.parameters():
                print(type(param), param.size())
            >>> <class 'torch.Tensor'> (20L,)
            >>> <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
            ```
        """
        for _name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        # TODO: docstring
        r"""Return an iterator over module parameters, yielding both the name of the 
            parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def buffers(self, recurse: bool = True) -> Iterator[Buffer]: # TODO doc
        """Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Returns:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Buffer]]: # TODO docstring
        r"""Return an iterator over module buffers, yielding both the name of the 
            buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Module"]:
        r"""Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for _name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""Return an iterator over immediate children modules, yielding 
            both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator["Module"]: # TODO DOC
        r"""Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        r"""Set the module in evaluation mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    def requires_pgrad_(self: T, requires_pgrad: bool = True) -> T: # TODO mudar
        r"""Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_pgrad_(requires_pgrad)
        return self

    def zero_pgrad(self, set_to_none: bool = True) -> None: # TODO isso é interessante mas vai mudar
        r"""Reset gradients of all model parameters.

        See similar function under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        for p in self.parameters():
            if p.pgrad is not None:
                if set_to_none:
                    p.pgrad = None
                else: # TODO revisar abaixo
                    if p.pgrad.grad_fn is not None:
                        p.pgrad.detach_()
                    else:
                        p.pgrad.requires_grad_(False)
                    p.pgrad.zero_()

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Return the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)
