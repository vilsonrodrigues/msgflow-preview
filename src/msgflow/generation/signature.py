import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    get_type_hints,
    GenericAlias,
)
import msgspec


class InputField:
    """
    Represents an input field in a model signature.

    Attributes:
        desc: A description of the input field. Defaults to an empty string.
    """

    def __init__(self, desc: Optional[str] = ""):
        self.desc = desc


class OutputField:
    """
    Represents an output field in a model signature.

    Attributes:
        desc: A description of the output field. Defaults to an empty string.
    """

    def __init__(self, desc: Optional[str] = ""):
        self.desc = desc


class _SignatureMeta(type):
    """
    Metaclass to process input and output fields in a model signature.

    This metaclass collects all `InputField` and `OutputField` instances defined in a class
    and stores them in `_inputs` and `_outputs` dictionaries, respectively.
    """

    def __new__(cls, name, bases, dct):
        inputs = {}
        outputs = {}

        for key, value in dct.items():
            if isinstance(value, InputField):
                inputs[key] = value
            elif isinstance(value, OutputField):
                outputs[key] = value

        dct["_inputs"] = inputs
        dct["_outputs"] = outputs
        return super().__new__(cls, name, bases, dct)


class Signature(metaclass=_SignatureMeta):
    """
    Base class for model signatures.

    This class provides functionality to define and inspect input and output fields
    of a model. It uses the `_SignatureMeta` metaclass to automatically collect
    `InputField` and `OutputField` instances.

    Example:
        ```python
        class CheckCitationFaithfulness(Signature):
            \"\"\"Verify that the text is based on the provided context.\"\"\"

            context: str = InputField(desc="Facts here are assumed to be true")
            text: str = InputField()
            faithfulness: bool = OutputField()
            evidence: dict[str, list[str]] = OutputField(
                desc="Supporting evidence for claims"
            )

        # Get the class docstring
        print(CheckCitationFaithfulness.get_instructions())
        # Output: "Verify that the text is based on the provided context."

        # Get the signature in string format
        print(CheckCitationFaithfulness.get_str_signature())
        # Output: "context: str, text: str -> faithfulness: bool, evidence: dict[str, list[str]]"

        # Get input descriptions
        print(CheckCitationFaithfulness.get_input_descriptions())
        # Output: [('context', 'str', 'Facts here are assumed to be true'), ('text', 'str', '')]

        # Get output descriptions
        print(CheckCitationFaithfulness.get_output_descriptions())
        # Output: [('faithfulness', 'bool', ''), ('evidence', 'dict[str, list[str]]', 'Supporting evidence for claims')]
        ```
    """

    @classmethod
    def _type_to_str(cls, type_obj: Any) -> str:
        """
        Converts a type object to a readable string representation.

        Args:
            type_obj: The type object to convert.

        Returns:
            A string representation of the type.
        """
        if isinstance(type_obj, GenericAlias):  # For generic types
            return str(type_obj)
        return type_obj.__name__

    @classmethod
    def _get_inputs(cls) -> Dict[str, str]:
        """
        Retrieves the input fields with their names and types.

        Returns:
            A dictionary mapping input field names to their types.
        """
        type_hints = get_type_hints(cls)
        return {key: cls._type_to_str(type_hints[key]) for key in cls._inputs}

    @classmethod
    def _get_outputs(cls) -> Dict[str, str]:
        """
        Retrieves the output fields with their names and types.

        Returns:
            A dictionary mapping output field names to their types.
        """
        type_hints = get_type_hints(cls)
        return {key: cls._type_to_str(type_hints[key]) for key in cls._outputs}

    @classmethod
    def get_str_signature(cls) -> str:
        """
        Returns the signature of the parameters in string format.

        Returns:
            A string representation of the input and output fields.
        """
        inputs = [f"{key}: {typ}" for key, typ in cls._get_inputs().items()]
        outputs = [f"{key}: {typ}" for key, typ in cls._get_outputs().items()]
        return ", ".join(inputs) + " -> " + ", ".join(outputs)

    @classmethod
    def get_input_descriptions(cls) -> List[Tuple[str, str, str]]:
        """
        Returns the descriptions and types of the input parameters.

        Returns:
            A list of tuples containing the input field name,
            type, and description.
        """
        inputs = cls._get_inputs()
        return [(key, typ, cls._inputs[key].desc) for key, typ in inputs.items()]

    @classmethod
    def get_output_descriptions(cls) -> List[Tuple[str, str, str]]:
        """
        Returns the descriptions and types of the output parameters.

        Returns:
            A list of tuples containing the output field name,
            type, and description.
        """
        outputs = cls._get_outputs()
        return [(key, typ, cls._outputs[key].desc) for key, typ in outputs.items()]

    @classmethod
    def get_instructions(cls) -> Optional[str]:
        """
        Returns the class docstring.

        Returns:
            The docstring of the class, or `None` if no docstring is present.
        """
        return cls.__doc__.strip() if cls.__doc__ else None

def parse_annotations(annotation: str) -> List[Tuple[str, str]]:
    """
    Parses a string of annotations in the format "field1: type1, field2: type2".
    Assumes `str` as the default type if none is provided.

    Used in `msgflow.signatures`.
    """
    # Remove unnecessary spaces and surrounding quotes, if any
    annotation = annotation.strip().strip('"')

    # Split pairs by leading commas (not inside brackets)
    pairs = re.split(r",\s*(?![^[]*\])", annotation)

    result = []
    for pair in pairs:
        # Separate key and type, assuming type is `str` if omitted
        if ":" in pair:
            key, value_type = map(str.strip, pair.split(":", 1))
        else:
            key, value_type = pair.strip(), "str"
        result.append((key, value_type))

    return result

def create_struct_from_signature(signature: str, struct_name: str = "DynamicStruct"):
    """
    Creates a msgspec struct class from a signature string.

    Args:
        signature: Signature string in the format "field1: type1, field2: type2".
        struct_name: Name of the struct class to create.

    Returns:
        A msgspec struct class
    """
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    annotations = parse_annotations(signature)

    # Prepare the list of fields for the msgspec
    struct_fields = []
    for name, type_str in annotations:
        # Support for built-in types
        if type_str in type_map:
            struct_fields.append((name, type_map[type_str.lower()]))
        else:
            # Try to handle more complex types
            try:
                struct_fields.append((name, eval(type_str)))
            except:
                raise ValueError(f"Unsupported type: `{type_str}`")

    # Create the struct class dynamically using defstruct
    DynamicStruct = msgspec.defstruct(struct_name, struct_fields)

    return DynamicStruct
