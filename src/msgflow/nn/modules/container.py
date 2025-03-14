import operator
from collections import abc as container_abcs, OrderedDict
from itertools import chain, islice
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Union,
    Tuple,
    TypeVar,
)
from typing_extensions import Self
from msgflow.message import Message
from msgflow.nn.modules.module import Module, _addindent


__all__ = [
    "Sequential",
    "ModuleList",
    "ModuleDict",
]

T = TypeVar("T", bound=Module)


class Sequential(Module):
    """A sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ==OrderedDict== of modules can be
    passed in. The ==forward()== method of ==Sequential== accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ==Sequential== provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ==Sequential== applies to each of the modules it stores (which are
    each a registered submodule of the ==Sequential==).

    What's the difference between a ==Sequential== and a
    ==msgflow.nn.ModuleList==? A ==ModuleList== is exactly what it
    sounds like--a list for storing ==Module== s! On the other hand,
    the layers in a ==Sequential== are connected in a cascading way.

    !!! example
        ``` py
        # TODO
        # Using Sequential to create a small model. When **model** is run,
        # input will first be passed to **Conv2d(1,20,5)**. The output of
        # **Conv2d(1,20,5)** will be used as the input to the first
        # **ReLU**; the output of the first **ReLU** will become the input
        # for **Conv2d(20,64,5)**. Finally, the output of
        # **Conv2d(20,64,5)** will be used as input to the second **ReLU**
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        ```
    """

    _modules: Dict[str, Module] = OrderedDict()

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, message: Union[str, Dict[str, Any], Message]):
        for module in self:
            message = module(message)
        return message

    def _get_mermaid(
        self,
        title: Optional[str] = None,  # TODO
        orientation: Optional[str] = "TD",
    ) -> str:
        mermaid_code = [
            "%%{",
            "        init: {",
            "            'theme': 'base',",
            "            'themeVariables': {",
            "            'primaryColor': '#E9E7E7',",
            "            'primaryTextColor': '#000000',",
            "            'primaryBorderColor': '#C0000',",
            "            'lineColor': '#F8B229',",
            "            'secondaryColor': '#91939C',",
            "            'tertiaryColor': '#fff'",
            "            }",
            "        }",
            "    }%%",
            f"flowchart {orientation}",
        ]

        mermaid_code.append("subgraph PARAMETERS")
        mermaid_code.append("direction LR")
        mermaid_code.append(f"param_msg([**msg**])")
        mermaid_code.append("end")

        first_node = None
        for i, module_name in enumerate(self._modules.keys()):
            node_id = f"node_{i}"
            if i == 0:
                first_node = node_id
            mermaid_code.append(f"{node_id}[ **msg = {module_name}﹙msg﹚** ]")

        for i in range(len(self._modules) - 1):
            mermaid_code.append(f"node_{i} --> node_{i + 1}")

        last_node = f"node_{len(self._modules) - 1}"
        mermaid_code.append(f"{last_node} --> node_return")
        mermaid_code.append("node_return([**return msg**])")

        if first_node:
            mermaid_code.append(f"PARAMETERS --> {first_node}")

        mermaid_code.append("%% Styles")
        mermaid_code.append(
            "classDef terminal fill:#FFF4DD,stroke:#333,stroke-width:2px;"
        )
        mermaid_code.append(
            "classDef parameter fill:#FFF4DD,stroke:#333,stroke-width:px;"
        )

        for i in range(len(self._modules)):
            mermaid_code.append(f"class node_{i} default;")
        mermaid_code.append("class node_return terminal;")

        mermaid_code.append(f"class param_msg parameter;")

        return "\n".join(mermaid_code)

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: Union[slice, int]) -> Union["Sequential", T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> "Sequential":
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> Self:
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of Sequential class, but {str(type(other))} is given."
            )

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> "Sequential":
        r"""Append a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> "Sequential":
        if not isinstance(module, Module):
            raise AssertionError(f"module should be of type: {Module}")
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(f"Index out of range: {index}")
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> "Sequential":
        for layer in sequential:
            self.append(layer)
        return self


class ModuleList(Module):
    """Holds submodules in a list.

    ==msgflow.nn.ModuleList== can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    ==torch.nn.Module== methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    
    !!! example

        # TODO
        ``` py
        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
        ```
    """

    _modules: Dict[str, Module]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """ Get the absolute index for the list of modules """
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "ModuleList"]:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)
 
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> Self:
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> "ModuleList":
        combined = ModuleList()
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    def __repr__(self):
        """Return a custom repr for ModuleList that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        """Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module: Module) -> "ModuleList":
        """Append a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def extend(self, modules: Iterable[Module]) -> Self:
        """Append modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleList.extend should be called with an "
                "iterable, but got " + type(modules).__name__
            )
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ModuleDict(Module):
    """Holds submodules in a dictionary.

    ==msgflow.nn.ModuleDict== can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    ==msgflow.nn.Module== methods.

    ==msgflow.nn.ModuleDict== is an **ordered** dictionary that respects

    * the order of insertion, and

    * in ==msgflow.nn.ModuleDict.update== the order of the merged
      ==OrderedDict==, ==dict== (started from Python 3.6) or another
      ==msgflow.nn.ModuleDict== (the argument to
      ==msgflow.nn.ModuleDict.update==).

    Note that ==msgflow.nn.ModuleDict.update== with other unordered mapping
    types (e.g., Python's plain ==dict== before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    !!! example

        # TODO
        ``` py    
        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
        ```
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """ Remove all items from the ModuleDict """
        self._modules.clear()
    
    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        """ Return an iterable of the ModuleDict keys """
        return self._modules.keys()

    def items(self) -> Iterable[Tuple[str, Module]]:
        """ Return an iterable of the ModuleDict key/value pairs """
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        """ Return an iterable of the ModuleDict values """
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        """Update the class **msgflow.nn.ModuleDict** with 
        key-value pairs from a mapping, overwriting existing keys.

        !!! note

            If ==modules== is an ==OrderedDict==, a ==msgflow.nn.ModuleDict==, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to ==msgflow.nn.Module==,
                or an iterable of key-value pairs of type (string, ==msgflow.nn.Module==)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(modules).__name__
            )

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]islice