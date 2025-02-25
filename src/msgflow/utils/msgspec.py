import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin
import msgspec


def get_literal_values(field_type: Any) -> Union[List[Any], None]:
    """
    Extracts the values from a `Literal` type.

    Args:
        field_type: The type to check for `Literal` values.

    Returns:
        A list of values if the type is a `Literal`, otherwise `None`.

    !!! example
        ```python
        literal_type = Literal["a", "b", "c"]
        values = get_literal_values(literal_type)
        print(values)  # Output: ["a", "b", "c"]
        ```
    """
    if get_origin(field_type) is Literal:
        return list(get_args(field_type))
    return None


def get_enum_values(field_type: Any) -> Union[Dict[str, Any], None]:
    """
    Extracts the values from an `Enum` type.

    Args:
        field_type: The type to check for `Enum` values.

    Returns:
        A dictionary mapping enum names to their values if the type is an `Enum`,
        otherwise `None`.

    !!! example
        ```python
        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        values = get_enum_values(Color)
        print(values)  # Output: {"RED": 1, "GREEN": 2, "BLUE": 3}
        ```
    """
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return {name: value.value for name, value in field_type.__members__.items()}
    return None


def get_field_type_name(field_type: Any) -> str:
    """
    Safely retrieves the name of a field type for JSON serialization.

    Args:
        field_type: The type to extract the name from.

    Returns:
        The name of the type or its string representation if it doesn't have a `__name__` attribute.

    !!! example
        ```python
        type_name = get_field_type_name(int)
        print(type_name)  # Output: "int"
        ```
    """
    if hasattr(field_type, "__name__"):
        return field_type.__name__
    return str(field_type)


def serialize_struct(cls: msgspec.Struct) -> Dict[str, Any]:
    """
    Extracts the definition of a `msgspec.Struct` class and returns a JSON-serializable dictionary.

    Supports basic types, nested `Struct`, `Literal`, and `Enum`.

    Args:
        cls: The `Struct` class to serialize.

    Returns:
        A dictionary containing the serialized definition of the `Struct`.

    !!! example
        ```python
        class Person(msgspec.Struct):
            name: str
            age: int

        serialized = serialize_struct(Person)
        print(serialized)
        # Output: {"name": "Person", "fields": [{"name": "name", "type": "str"}, {"name": "age", "type": "int"}]}
        ```
    """
    definition = {"name": cls.__name__, "fields": []}
    
    for field_name, field_type in cls.__annotations__.items():
        field_def = {"name": field_name}
        
        # Handle default value safely for JSON
        if field_name in cls.__dict__:
            default_value = cls.__dict__[field_name]
            try:
                msgspec.json.encode(default_value)
                field_def["default"] = default_value
            except TypeError:
                if isinstance(default_value, Enum):
                    field_def["default"] = default_value.value
                else:
                    field_def["default"] = str(default_value)
        
        # Check if it's a Literal type
        literal_values = get_literal_values(field_type)
        if literal_values is not None:
            field_def["type"] = "Literal"
            field_def["values"] = literal_values
        
        # Check if it's an Enum
        elif enum_values := get_enum_values(field_type):
            field_def["type"] = "Enum"
            field_def["values"] = enum_values
        
        # Check if it's a nested Struct
        elif isinstance(field_type, type) and issubclass(field_type, msgspec.Struct):
            field_def["type"] = serialize_struct(field_type)
        
        # Basic type
        else:
            field_def["type"] = get_field_type_name(field_type)
            
        definition["fields"].append(field_def)
    
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

