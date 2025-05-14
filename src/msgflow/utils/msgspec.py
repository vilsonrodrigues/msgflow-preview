import inspect
import os
from enum import Enum
from uuid import UUID, uuid4
from typing import Any, Dict, List, Literal, Optional, Union, Type, get_args, get_origin, get_type_hints
import msgspec
from typing_extensions import Generic, TypeVar


_NoneType = type(None)

def _serialize_type(type_hint: Any) -> Any:
    """Recursively serializes a type hint to a JSON-safe structure."""
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Union:
        # Treat Optional[X] as Union[X, NoneType]
        union_args = [_serialize_type(arg) for arg in args]
        if len(union_args) == 2 and _serialize_type(_NoneType) in union_args:
            inner_type = next(t for t in union_args if t != _serialize_type(_NoneType))
            return {"kind": "optional", "type": inner_type}
        else:
            return {"kind": "union", "types": union_args}
    elif origin is list or origin is List:
        if not args:
            # List with no arguments, as in List = []
            return {"kind": "list", "type": _serialize_type(Any)}
        return {"kind": "list", "type": _serialize_type(args[0])}
    elif origin is dict or origin is Dict:
        if not args:
             return {"kind": "dict", "key_type": _serialize_type(Any), "value_type": _serialize_type(Any)}
        return {"kind": "dict", "key_type": _serialize_type(args[0]), "value_type": _serialize_type(args[1])}
    elif origin is Literal:
        # Ensure literal values ​​are JSON-serializable
        literal_values = []
        for val in args:
            try:
                msgspec.json.encode(val)
                literal_values.append(val)
            except TypeError:
                literal_values.append(str(val)) # Fallback to string if not serializable
        return {"kind": "literal", "values": literal_values}
    elif isinstance(type_hint, type):
        if issubclass(type_hint, Enum):
            # Serialize Enum by name and values ​​(value must be JSON-safe)
            enum_values = {}
            for member in type_hint:
                try:
                    msgspec.json.encode(member.value)
                    enum_values[member.name] = member.value
                except TypeError:
                     enum_values[member.name] = str(member.value) # Fallback
            return {"kind": "enum", "name": type_hint.__name__, "values": enum_values}
        elif issubclass(type_hint, msgspec.Struct):
            return {"kind": "struct", "name": type_hint.__name__, "definition": serialize_struct(type_hint)}
        elif type_hint is Any:
             return {"kind": "any"}
        elif type_hint is _NoneType:
             return {"kind": "none"}
        elif type_hint is UUID:
             return {"kind": "base", "name": "UUID"}
        else:            
            return {"kind": "base", "name": type_hint.__name__} # Base types (int, str, float, bool)
    elif isinstance(type_hint, TypeVar):
        # Tratar TypeVar - serializar seu nome e possivelmente seu default
        # O default do TypeVar em si pode não ser diretamente útil sem contexto.
        # default_type = getattr(type_hint, '__default__', None) # type_hint.__default__ não existe
        # type_hint.__bound__ ou type_hint.__constraints__ podem ser relevantes
        # Por simplicidade, apenas serializamos o nome.
        return {"kind": "typevar", "name": type_hint.__name__}
    else:        
        return {"kind": "unknown", "repr": str(type_hint)} # Fallback for unknown types

def serialize_struct(cls: Type[msgspec.Struct]) -> Dict[str, Any]:
    """
    Extracts the definition of a `msgspec.Struct` class and returns a JSON-serializable 
    dictionary. Supports basic, nested types `Struct`, `Literal`, `Enum`, `Optional`, 
    `List`, `Dict`, `Union`, `UUID`.
    """
    if not isinstance(cls, type) or not issubclass(cls, msgspec.Struct):
        raise TypeError("Input must be a msgspec.Struct class")

    definition = {"name": cls.__name__, "fields": []}
    annotations = get_type_hints(cls) # Use get_type_hints to resolve forward refs

    # Usar msgspec.inspect para obter informações dos campos de forma confiável
    #try:
    #    fields_info = msgspec.inspect.fields(cls)
    #except TypeError:
         # Pode falhar se a Struct não for "concreta" (e.g., ainda genérica)
         # Tentativa de fallback, mas pode ser impreciso
    fields_info = [] # Ou alguma outra lógica de fallback

    field_info_map = {f.name: f for f in fields_info}

    for field_name, field_type in annotations.items():
        if field_name.startswith('_') and field_name.endswith('_'): # Ignora campos privados/especiais como __slots__
            continue

        field_def = {"name": field_name}

        # Serializar o tipo
        field_def["type"] = _serialize_type(field_type)

        # Obter valor padrão do field_info se disponível
        field_info = field_info_map.get(field_name)
        if field_info and field_info.default is not msgspec.NODEFAULT:
            default_value = field_info.default
            try:
                # Tentar serializar diretamente se for JSON-compatível
                msgspec.json.encode(default_value)
                field_def["default"] = default_value
            except TypeError:
                # Se não for JSON-compatível, tentar casos específicos
                if isinstance(default_value, Enum):
                    field_def["default"] = default_value.value # Usar valor do Enum
                elif isinstance(default_value, UUID):
                     field_def["default"] = str(default_value) # Serializar UUID como string
                # Adicionar outros casos conforme necessário (e.g., datetime)
                else:
                    # Último recurso: converter para string (pode perder informação)
                     # Evitar serializar callables (como default_factory) diretamente
                    if not callable(default_value):
                        field_def["default"] = str(default_value)
                    else:
                         # Não serializar fábricas de padrão por enquanto
                         pass # ou field_def["default_factory"] = True

        # Tratar default_factory
        if field_info and field_info.default_factory is not msgspec.NODEFAULT:
             # Marcar que existe uma fábrica, mas não serializar a função em si
             field_def["has_default_factory"] = True
             # Se a fábrica for uuid4, podemos serializar isso especificamente
             if field_info.default_factory is uuid4:
                 field_def["default_factory_name"] = "uuid4"


        definition["fields"].append(field_def)

    # Adicionar informação sobre kw_only se relevante
    sig = inspect.signature(cls.__init__)
    kw_only_params = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    if kw_only_params:
        definition["kw_only_fields"] = kw_only_params


    # Informação sobre Generic[T] - experimental
    if hasattr(cls, '__orig_bases__'):
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is Generic:
                args = get_args(base)
                definition['generic_params'] = [_serialize_type(arg) for arg in args]


    return definition

def serialize_msgspec_struct(cls: msgspec.Struct) -> Dict[str, Any]:
    """
    Serializes a `msgspec.Struct` into a generation schema dictionary.

    Args:
        cls: The `Struct` class to serialize.

    Returns:
        A dictionary containing the serialized schema.

    !!! example
        ```python
        class Person(msgspec.Struct):
            name: str
            age: int

        schema = serialize_msgspec_struct(Person)
        print(schema)
        # Output: {"msgflow_type": "generation_schema", "provider": "msgspec", "instance_type": "struct", "state": {...}}
        ```
    """
    data = {
        "msgflow_type": "generation_schema",
        "provider": "msgspec",
        "instance_type": "struct"
    }
    state = serialize_struct(cls)                  
    data["state"] = state
    return data     


def create_enum_class(name: str, values: Dict[str, Any]) -> type:
    """
    Dynamically creates an `Enum` class with the provided values.

    Args:
        name: The name of the `Enum` class.
        values: A dictionary mapping enum names to their values.

    Returns:
        The dynamically created `Enum` class.

    !!! example
        ```python
        enum_class = create_enum_class("Color", {"RED": 1, "GREEN": 2, "BLUE": 3})
        print(enum_class.RED)  # Output: <Color.RED: 1>
        ```
    """
    return Enum(name, values)


def deserialize_struct(definition: Dict[str, Any], type_mapping: Dict[str, type] = None) -> type:
    """
    Reconstructs a `Struct` class from its serialized definition.

    Supports basic types, nested `Struct`, `Literal`, and `Enum`.

    Args:
        definition: The serialized definition of the `Struct`.
        type_mapping: A dictionary mapping type names to their corresponding types.
            Defaults to basic types like `int`, `str`, `float`, etc.

    Returns:
        type: The reconstructed `Struct` class.

    Example:
        ```python
        definition = {
            "name": "Person",
            "fields": [
                {"name": "name", "type": "str"},
                {"name": "age", "type": "int"}
            ]
        }
        Person = deserialize_struct(definition)
        print(Person(name="Alice", age=30))  # Output: Person(name="Alice", age=30)
        ```
    """
    if type_mapping is None:
        type_mapping = {
            "int": int,
            "str": str,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict
        }

    fields: List[Union[tuple[str, type], tuple[str, type, Any]]] = []
    
    for field_def in definition["fields"]:
        fname = field_def["name"]
        ftype = field_def["type"]
        
        if isinstance(ftype, dict):
            actual_type = deserialize_struct(ftype, type_mapping)
        elif ftype == "Literal":
            values = field_def["values"]
            actual_type = Literal[tuple(values)]  # type: ignore
        elif ftype == "Enum":
            enum_name = f"{definition['name']}_{fname}_Enum"
            actual_type = create_enum_class(enum_name, field_def["values"])
        else:
            actual_type = type_mapping.get(ftype, str)
        
        if "default" in field_def:
            fields.append((fname, actual_type, field_def["default"]))
        else:
            fields.append((fname, actual_type))
    
    return msgspec.defstruct(definition["name"], fields)

def export_to_toml(obj, filepath):
    with open(filepath, "wb") as f:
        f.write(msgspec.toml.encode(obj))    

def export_to_json(obj, filepath, indent=4):
    with open(filepath, "wb") as f:
        obj_b = msgspec.json.encode(obj)
        formatted_obj_b = msgspec.json.format(obj_b, indent=indent)
        f.write(formatted_obj_b)

def save(obj: object, f: Union[str, os.PathLike], format: Optional[Literal["toml", "json"]] = "toml"):
    """
    Save a Python object to a file in either TOML or JSON format.

    Args:
        data: Saved object
        filepath: A string or os.PathLike object containing a file name
        format: The format to save the file in. Can be "toml" or "json", defaults to "toml"

    Raises:
        ValueError: If the provided format is not "toml" or "json"
        FileNotFoundError: If the directory of the provided filepath does not exist

    !!! example
        ``` python
        data = {"name": "Satoshi", "age": 42}
        save(data, "output", format="toml")
        save(data, "output", format="json")
        ```
    """
    directory = os.path.dirname(f)
    if directory and not os.path.exists(directory):
        raise FileNotFoundError(f"The directory `{directory}` does not exist")
    
    if format == "toml":
        export_to_toml(obj, f)
    elif format == "json":
        export_to_json(obj, f)
    else:
        raise ValueError(f"Unsupported format: `{format}`. Use `toml` or `json`")

def read_json(filepath):
    with open(filepath, "rb") as f:
        return msgspec.json.decode(f.read())

def read_toml(filepath):
    with open(filepath, "rb") as f:
        return msgspec.toml.decode(f.read())

def load(f: Union[str, os.PathLike]) -> Any:
    """
    Load data from a file in either JSON or TOML format.

    Args:
        f: A string or os.PathLike object containing a file name

    Returns:
        The Python object loaded from the file

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file extension is not ".json" or ".toml"

    !!! example
        ``` python
        data = load("data.toml")
        data = load("data.json")
        ```        
    """
    if not os.path.exists(f):
        raise FileNotFoundError(f"The file `{f}` does not exist.")

    if f.endswith(".json"):
        return read_json(f)
    elif f.endswith(".toml"):
        return read_toml(f)
    else:
        raise ValueError(f"Unsupported file extension: `{f}`. Use `.json` or `.toml`")

def struct_to_dict(obj):
    """
    Recursively converts a msgspec.Struct object to a pure Python dictionary
    """
    if isinstance(obj, msgspec.Struct):
        # Convert the struct to a dictionary and recursively process each value
        return {k: struct_to_dict(v) for k, v in msgspec.structs.asdict(obj).items()}
    elif isinstance(obj, list):
        # Convert each item in the list recursively
        return [struct_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        # If it is a dictionary, recursively convert its values
        return {k: struct_to_dict(v) for k, v in obj.items()}
    else:
        # Returns the value as is for simple types
        return obj

