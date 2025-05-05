import ast
import re
from typing import (
    Any,
    Dict,
    GenericAlias,    
    List,
    Literal,
    Optional,
    Set,
    Union,    
    Tuple,
    Type,
    get_args,
    get_origin,
    get_type_hints,    
)
import msgspec
from msgflow.generation.reasoning.cot import ChainOfThoughts, COT_SYSTEM_MESSAGE
from msgflow.generation.reasoning.react import ReAct, REACT_SYSTEM_MESSAGE
from msgflow.generation.reasoning.self_consistency import SelfConsistency, SELF_CONSISTENCY_SYSTEM_MESSAGE
from msgflow.generation.reasoning.tot import TreeOfThoughts, TOT_SYSTEM_MESSAGE
from msgflow.logger import logger
from msgflow.utils.chat import apply_xml_tags


SIGNATURE_SYSTEM_MESSAGES = {
    ChainOfThoughts: COT_SYSTEM_MESSAGE,
    ReAct: REACT_SYSTEM_MESSAGE,
    SelfConsistency: SELF_CONSISTENCY_SYSTEM_MESSAGE,
    TreeOfThoughts: TOT_SYSTEM_MESSAGE
}


SIGNATURE_DEFAULT_SYSTEM_MESSAGE = """
Your goal is to provide accurate, helpful, and well-reasoned responses.
Carefully analyze the user's request to fully understand the objective. Address all parts of the query.
Think step-by-step to formulate your answer. Where appropriate, briefly explain your reasoning process.
Structure your response clearly and concisely using dicts, lists, or other formatting.
Strive for factual accuracy.
Be helpful and informative, focusing on directly answering the user's prompt.
""".strip()

class InputField:
    """
    Represents an input field in a model signature.

    Attributes:
        desc: A description of the input field. Defaults to an empty string.
        ex: Input parameter example.
    """

    def __init__(self, desc: Optional[str] = None, ex: Optional[str] = None):
        self.desc = desc
        self.ex = ex


class OutputField:
    """
    Represents an output field in a model signature.

    Attributes:
        desc: A description of the output field. Defaults to an empty string.
        ex: Output parameter example.
    """

    def __init__(self, desc: Optional[str] = None, ex: Optional[str] = None):
        self.desc = desc
        self.ex = ex
        

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
    # TODO: tem bugs no parser quando usa optional

    @classmethod
    def get_input_descriptions(cls) -> List[Tuple[str, str, str]]:
        """
        Returns the descriptions and types of the input parameters.

        Returns:
            A list of tuples containing the input field name,
            type, and description.
        """
        inputs = cls._get_inputs()
        return [(key, typ, cls._inputs[key].desc, cls._inputs[key].ex) 
                for key, typ in inputs.items()]

    @classmethod
    def get_output_descriptions(cls) -> List[Tuple[str, str, str]]:
        """
        Returns the descriptions and types of the output parameters.

        Returns:
            A list of tuples containing the output field name,
            type, and description.
        """
        outputs = cls._get_outputs()
        return [(key, typ, cls._outputs[key].desc, cls._outputs[key].ex) 
                for key, typ in outputs.items()]

    @classmethod
    def get_instructions(cls) -> Optional[str]:
        """
        Returns the class docstring.

        Returns:
            The docstring of the class, or `None` if no docstring is present.
        """
        return cls.__doc__.strip() if cls.__doc__ else None

    @classmethod
    def get_input_examples(cls) -> Dict[str, Optional[str]]:
        """
        Returns a mapping of input field names to their examples.

        Returns:
            A dictionary where keys are input field names and values are
            their corresponding examples ('ex'). If an example is not
            provided for a field, its value will be None.
        """
        return {key: field.ex for key, field in cls._inputs.items()}

    @classmethod
    def get_output_examples(cls) -> Dict[str, Optional[str]]:
        """
        Returns a mapping of output field names to their examples.

        Returns:
            A dictionary where keys are output field names and values are
            their corresponding examples ('ex'). If an example is not
            provided for a field, its value will be None.
        """
        return {key: field.ex for key, field in cls._outputs.items()}


ALLOWED_TYPES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": List,
    "dict": Dict,
    "literal": Literal,
    "any": Any,
    "union": Union,
    "optional": Optional,
    "tuple": Tuple,
    "none": type(None),
    "set": Set
}

def _parse_literal_args(args_str: str) -> tuple:
    """Safely parse arguments inside Literal[...] using ast."""
    try:
        # Wrap the string in square brackets to parse it as a literal list
        parsed_node = ast.parse(f"[{args_str}]", mode="eval")
        # Extract node values ​​from the parsed literal list
        # ast.literal_eval safely evaluates strings, numbers, tuples, lists, dicts, bools, None
        values = [ast.literal_eval(node) for node in parsed_node.body.elts]
        return tuple(values)
    except (SyntaxError, ValueError, TypeError) as e:
        logger.error(str(e))
        raise ValueError(f"Invalid literal arguments: `{args_str}`")
    

def _split_args(args_str: str) -> list[str]:
    """
    Splits a string of arguments (e.g. "str, list[str], Literal['a', 'b']")
    respecting nested brackets and basic single/double quotes.
    """
    args = []
    level = 0  # Bracket/parentheses nesting level
    current_arg_start = 0
    in_quotes = None # Controls whether it is enclosed in single or double quotes

    # Handle the special case of empty arguments as in Tuple[()]
    if not args_str.strip():
        return []

    for i, char in enumerate(args_str):
        if char in ("[", "{", "(") and not in_quotes:
            level += 1
        elif char in ("]", "}", ")") and not in_quotes:
            level -= 1
        elif char == "'" or char == '"':
            if in_quotes == char:
                in_quotes = None # Got out of the quotes
            elif in_quotes is None:
                in_quotes = char # Entered the quotes

        # Divide by comma only if not nested and not inside quotes
        elif char == "," and level == 0 and not in_quotes:
            args.append(args_str[current_arg_start:i].strip())
            current_arg_start = i + 1

    # Add the last argument (or the only argument)
    args.append(args_str[current_arg_start:].strip())
    return [arg for arg in args if arg] # Filter empty strings

def _parse_type_string(type_str: str) -> type:
    """
    Recursively parses a type string (e.g. "dict[str, list[str]]")
    into a real Python type object, using ALLOWED_TYPES (case-insensitive).
    """
    type_str = type_str.strip()

    if not type_str:
        raise ValueError("The type string cannot be empty.")

    type_str_lower = type_str.lower() # Convert to lowercase for lookup

    # Base case: Simple types (str, int, bool, any, none)
    # Checks if it is NOT a generic type that needs argument parsing
    simple_types_lower = {"str", "int", "float", "bool", "any", "none"}
    if type_str_lower in simple_types_lower and type_str_lower in ALLOWED_TYPES:
         return ALLOWED_TYPES[type_str_lower]

    # Recursive case: Generic types like list[T], dict[K, V], Union[X, Y], 
    # Literal[...], Optional[T], Tuple[...]
    # Use regex to find the pattern TypeName[arguments]
    # Type name (group 1) can be case-insensitive now
    # re.DOTALL to span multiple lines in args    
    match = re.match(r"^\s*(\w+)\s*\[(.*)\]\s*$", type_str, re.DOTALL)
    if match:
        base_type_name, args_str = match.groups()
        base_type_name = base_type_name.strip()
        args_str = args_str.strip() # Internal arguments

        base_type_name_lower = base_type_name.lower() # Convert to lowercase for lookup

        if base_type_name_lower not in ALLOWED_TYPES:
            # Use the original name in the error for clarity
            raise ValueError(f"Base type not supported: `{base_type_name}` in `{type_str}`")

        base_type = ALLOWED_TYPES[base_type_name_lower] # Search using lowercase version

        if base_type is Literal:
            try:
                parsed_args = _parse_literal_args(args_str)
                if not parsed_args:
                     raise ValueError("Literal[...] cannot be empty")
                return Literal[parsed_args]
            except Exception as e:
                 raise ValueError(f"Failed to parse Literal arguments `{args_str}`: {e}")

        elif base_type is Optional:
            arg_strs_list = _split_args(args_str)
            if len(arg_strs_list) != 1:
                raise ValueError("Optional[...] requires exactly 1 argument, "
                                 f"got {len(arg_strs_list)} in `{type_str}`")
            inner_type = _parse_type_string(arg_strs_list[0])
            return Union[inner_type, type(None)]

        elif base_type in (List, Dict, Union, Tuple):
             arg_strs_list = _split_args(args_str)

             if base_type is Tuple:
                 if not arg_strs_list:
                     return Tuple[()]
                 if len(arg_strs_list) == 2 and arg_strs_list[1] == "...":
                     item_type = _parse_type_string(arg_strs_list[0])
                     return Tuple[item_type, ...]

             if not arg_strs_list and base_type is not Tuple:
                 # Use original name in error
                 raise ValueError(f"{base_type_name}[...] must contain arguments.")

             parsed_args = tuple(_parse_type_string(arg) for arg in arg_strs_list)

             if base_type is Dict and len(parsed_args) != 2:
                 raise ValueError("Dict requires exactly 2 arguments (key, value), "
                                  f"got {len(parsed_args)} in `{type_str}`")
             if base_type is List and len(parsed_args) != 1:
                 raise ValueError("List requires exactly 1 argument, got "
                                  f"{len(parsed_args)} in `{type_str}`")
             if base_type is Union:
                  if len(parsed_args) == 0:
                      raise ValueError(f"Union[...] cannot be empty in `{type_str}`")
                  if len(parsed_args) == 1:
                      return parsed_args[0] # Union[T] simplify to T

             try:
                 return base_type[parsed_args]
             except TypeError as e:
                 raise ValueError(f"Invalid type arguments for {base_type_name}: "
                                  f"{parsed_args}. Error: {e}")
        else:
             raise ValueError(f"The construction of the generic type `{base_type_name}`"
                              " is not implemented or is invalid.")

    # If it is not a simple type and does not match the generic pattern TypeName[...]
    # Check again if it is a permitted simple type (using lowercase)
    if type_str_lower in ALLOWED_TYPES:
        return ALLOWED_TYPES[type_str_lower]

    raise ValueError(f"Unsupported or malformed type string: c{type_str}`")

def parse_annotations(signature: str) -> List[Tuple[str, str]]:
    """
    Parses a signature string in the format "field1:type1, field2:type2, field3".
    Assumes `str` as default type if omitted.
    Handles nested types and literals with commas.
    """
    fields = []
    current_pos = 0
    level = 0 
    in_quotes = None
    current_field_start = 0
    signature = signature.strip()

    if not signature:
        return []

    while current_pos < len(signature):
        char = signature[current_pos]

        if char in ("[", "{", "(") and not in_quotes:
            level += 1
        elif char in ("]", "}", ")") and not in_quotes:
            level -= 1
            if level < 0:
                 raise ValueError("Unbalanced bracket/parentheses "
                                  f"nesting near `{signature[current_pos:]}`")
        elif char == "'" or char == '"':
             if in_quotes == char:
                 in_quotes = None
             elif in_quotes is None:
                 in_quotes = char

        # Divide by comma only if not nested and not inside quotes
        if char == "," and level == 0 and not in_quotes:
            field_str = signature[current_field_start:current_pos].strip()
            if field_str: # Avoid adding empty strings if there are extra commas
                fields.append(field_str)
            current_field_start = current_pos + 1

        current_pos += 1

    if level != 0:
        raise ValueError("Unbalanced bracket/parentheses nesting in signature.")
    if in_quotes:
        raise ValueError("Unclosed quotation marks in signature.")

    last_field_str = signature[current_field_start:].strip()
    if last_field_str:
        fields.append(last_field_str)

    result = []
    for field_str in fields:
        parts = field_str.split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value_type = parts[1].strip()
            if not key:
                 raise ValueError(f"Field name cannot be empty in`{field_str}`")
            if not value_type:
                 raise ValueError(f"Type cannot be empty after ':' in `{field_str}`")
        else:
            key = field_str.strip()
            value_type = "str" # Default type
            if not key:
                 raise ValueError(f"Field name cannot be empty in `{field_str}`")

        result.append((key, value_type))

    return result

def create_struct_from_str_signature(signature: str, struct_name: Optional[str] = "DynamicStruct"):
    """
    Creates a struct msgspec class from a signature string,
    using a type-safe and case-insensitive parser for type names.

    Args:
        signature: Signature string (e.g. "field1: type1, field2: type2").
        struct_name: Name of the struct class to create.

    Returns:
        A struct msgspec class.

    Raises:
        ValueError: If the signature or types are invalid/unsupported.
        RuntimeError: For unexpected errors during parsing.
    """
    annotations = parse_annotations(signature) # Parse the signature into (name, string_type) pairs

    struct_fields = []
    for name, type_str in annotations:
        try:            
            parsed_type = _parse_type_string(type_str)
            struct_fields.append((name, parsed_type))
        except ValueError as e:            
            raise ValueError(f"Error parsing type for field `{name}` (type='{type_str}'): {e}")
        except Exception as e: # Catch other unexpected errors
            raise RuntimeError(f"Unexpected error parsing type `{type_str}` to field `{name}`: {e}")

    # Checks if any field was actually parsed if the signature was not empty
    if not struct_fields and signature.strip():
        raise ValueError("Unable to parse any fields in the provided signature.")

    try:
        DynamicStruct = msgspec.defstruct(struct_name, struct_fields)
    except Exception as e:
        raise RuntimeError(f"Error creating msgspec.defstruct `{struct_name}`: {e}")

    return DynamicStruct

def _parse_example_str(example_str: str, target_type: Type) -> Any:
    """
    Attempts to parse an example string into the target Python type.
    Uses ast.literal_eval for safety and to handle Python literals.
    """
    origin_type = get_origin(target_type)
    
    # Trata 'None' literal -> None Python
    if example_str.strip().lower() == "none":
        # Checks if the target type allows None (Optional)
        is_optional = origin_type is Union and type(None) in get_args(target_type)
        if target_type is type(None) or is_optional:
             return None
        else:
             raise ValueError(f"Received string `None` but target type `{target_type}` "
                              " is not None or Optional.")

    # Analyze based on destination type
    if target_type is str:
        return example_str
    elif target_type is bool:
        low_ex = example_str.strip().lower()
        if low_ex in ("true", "yes", "1"):
            return True
        elif low_ex in ("false", "no", "0"):
            return False
        else:
            raise ValueError(f"Could not parse `{example_str}` as boolean.")
    elif target_type is int:
        return int(example_str)
    elif target_type is float:
        return float(example_str)
    elif origin_type in (list, dict, tuple, set, List, Dict, Tuple, Set):
        try:
            parsed_value = ast.literal_eval(example_str)
            expected_type = list if origin_type in (list, List) else \
                            dict if origin_type in (dict, Dict) else \
                            tuple if origin_type in (tuple, Tuple) else \
                            set if origin_type in (set, Set) else None
            
            if expected_type and not isinstance(parsed_value, expected_type):
                 raise TypeError(f"Parsed value `{parsed_value}` is not of "
                                 "expected type {expected_type} for {target_type}")
                 
            return parsed_value
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
            raise ValueError(f"Failed to parse `{example_str}` as {target_type} using ast.literal_eval: {e}") from e
    else:
        try:
            # It might be useful if the type is, for example, Optional[int] and the example is "123"
            return ast.literal_eval(example_str)
        except (ValueError, SyntaxError, TypeError, MemoryError):
             raise ValueError(f"Unsupported type `{target_type}` for automatic parsing of example string '{example_str}'")


def get_examples_from_signature(
    signature_cls: Type[Signature]
) -> Optional[Tuple[Dict[str, str], str]]:
    """
    Processes examples of a Signature class, returning a dict for inputs
    and a JSON string for outputs, only if all examples are present.

    Args:
        signature_cls: The Signature class (which inherits from Signature) to process.

    Returns:
        A tuple containing:
        - Dictionary mapping input names to their example strings.
        - JSON string mapping output names to their *parsed* examples.
        Returns None if any field (input or output) does not have an example defined.
        Raises ValueError/TypeError if an error occurs in parsing the output examples
        or in the JSON encoding.
    """
    input_descs = signature_cls.get_input_descriptions()
    output_descs = signature_cls.get_output_descriptions()

    # Check if ALL fields have an instance (not None)
    all_inputs_have_examples = all(desc[3] is not None for desc in input_descs)
    all_outputs_have_examples = all(desc[3] is not None for desc in output_descs)

    if not (all_inputs_have_examples and all_outputs_have_examples):
        return None

    # 1. Create dictionary of input examples (name -> example string)
    # Ensured that desc[3] (example) is not None because of the above check
    input_examples_dict = {desc[0]: desc[3] for desc in input_descs} 

    # 2. Create the dictionary of output examples (name -> *parsed* example)
    output_parsed_dict = {}
    type_hints = get_type_hints(signature_cls) # Get the actual types

    try:
        for name, _, _, example_str in output_descs:
            target_type = type_hints.get(name)
            if target_type is None:
                raise ValueError(f"Type hint not found for field "
                                 f"output `{name}` em {signature_cls.__name__}")
             
            # Use the helper function to parse the example string to the correct type
            parsed_value = _parse_example_str(example_str, target_type) 
            output_parsed_dict[name] = parsed_value

    except (ValueError, TypeError) as e:
        raise ValueError(f"Error parsing output examples for "
                         f"{signature_cls.__name__}: {e}") from e
    
    try: # 3. Encode the parsed output dictionary to a JSON string
        output_json_string = msgspec.json.encode(output_parsed_dict)
    except TypeError as e:
        raise TypeError(f"Error encoding parsed outputs to JSON in "
                        f"{signature_cls.__name__}: {e}") from e

    return input_examples_dict, output_json_string

def get_expected_output_from_signature(
    inputs_desc: List[Tuple[str, str, str, Union[str, None]]],
    outputs_desc: List[Tuple[str, str, str, Union[str, None]]]
) -> str:
    expected_output = "Your task inputs are:\n\n"
    for i, input_desc in enumerate(inputs_desc, 1):
        part = f"{i}. `{input_desc[0]}` ({input_desc[1]})"
        if len(input_desc) == 3 and input_desc[2] is not None:
            part += f": {input_desc[2]}"
        expected_output += part + "\n"
    expected_output += "\nYour final answer should have:\n\n"
    for i, output_desc in enumerate(outputs_desc, 1):
        part = f"{i}. `{output_desc[0]}` ({output_desc[1]})"
        if len(output_desc) == 3 and output_desc[2] is not None:
            part += f": {output_desc[2]}" 
        expected_output += f"{part}\n"
    expected_output += "\nBe consise in choosing your answers. Write an encoded JSON."
    return expected_output

def get_task_template_from_signature(
    inputs_desc: List[Tuple[str, str, str, Union[str, None]]]
) -> str:
    task_template = ""
    for input_desc in inputs_desc:
        part = apply_xml_tags(input_desc[0], f"{{{{ {input_desc[0]} }}}}")
        task_template += part + "\n"
    task_template = task_template.strip()
    return task_template
