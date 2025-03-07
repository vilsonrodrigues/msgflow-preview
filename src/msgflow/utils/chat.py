import re
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Union,
    get_origin,
)
from jinja2 import Template


def adapt_struct_schema_to_json_schema(
    original_schema: Dict[str, Any],
) -> Dict[str, Any]:
    def resolve_ref(ref: str, defs: Dict) -> Dict:
        """Resolves a reference `$ref` using the dictionary `$defs`"""
        ref_key = ref.split("/")[-1]
        return defs.get(ref_key, {})

    root_ref = original_schema.get("$ref", "")
    defs = original_schema.get("$defs", {})

    root_schema = resolve_ref(root_ref, defs)

    def deep_resolve_and_enforce_properties(schema: Dict) -> Dict:
        if "$ref" in schema:
            schema = resolve_ref(schema["$ref"], defs)

        # Enforce additionalProperties: false for all object types
        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        if "properties" in schema:
            schema["properties"] = {
                k: deep_resolve_and_enforce_properties(v)
                for k, v in schema["properties"].items()
            }

        if "items" in schema:
            schema["items"] = deep_resolve_and_enforce_properties(schema["items"])

        return schema

    resolved_schema = deep_resolve_and_enforce_properties(root_schema)

    adapted_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": root_schema.get("title", "response").lower(),
            "schema": {
                "type": resolved_schema.get("type", "object"),
                "properties": resolved_schema["properties"],
                "required": resolved_schema.get("required", []),
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    return adapted_schema


def chatml_to_steps_format(model_state, response):
    steps = []
    pending_tool_calls = {}

    for message in model_state:
        if message["role"] == "user" and "content" in message:
            steps.append({"task": message["content"]})

        elif message["role"] == "assistant" and "content" in message:
            steps.append({"assistant": message["content"]})

        elif message.get("tool_calls"):
            # Iterates over all function calls in the `tool_calls` list
            for tool_call in message["tool_calls"]:
                fn_call_entry = {
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                    "resuls": None, # To be updated when the answer is found
                }
                # Add each function call separately
                steps.append({"tool_call": fn_call_entry})
                pending_tool_calls[tool_call["id"]] = fn_call_entry

        elif message["role"] == "tool" and message.get("tool_call_id"):
            # Check if there is a corresponding function call pending
            tool_call_id = message["tool_call_id"]
            if tool_call_id in pending_tool_calls:
                # Update the result of the corresponding function call
                pending_tool_calls[tool_call_id]["result"] = message.get("content", "")

    if response:
        steps["assistant"] = response

    return steps

def convert_camel_to_snake_case(camel_str) -> str:
    snake_str = re.sub(r"(?<!^)([A-Z])", r"_\1", camel_str).lower()
    return snake_str

def text_code_to_callable(text_code: str) -> Callable:
    """Convert text Python code to a callable object"""
    local_context = {}
    global_context = globals()
    exec(text_code, global_context, local_context)
    module_name = list(local_context.keys())[0]
    module = local_context[module_name]
    return module

def clean_docstring(docstring: str) -> str:
    """
    Cleans the docstring by removing the Args section.

    Args:
        docstring: Complete docstring to clean

    Returns:
        Clean docstring without Args section
    """
    if not docstring:
        return ""

    # Remove the Args section and any text after it
    cleaned = re.sub(r"\s*Args:.*", "", docstring, flags=re.DOTALL).strip()

    return cleaned


def parse_docstring_args(docstring: str) -> Dict[str, str]:
    """
    Extracts parameter descriptions from the Args section of the docstring.

    Args:
        docstring: Complete docstring of the function/class

    Returns:
        Dictionary with parameter descriptions
    """
    if not docstring:
        return {}

    # Find the Args section
    args_match = re.search(
        r"Args:\s*(.*?)(?:\n\n|\n[A-Za-z]+:|\Z)", docstring, re.DOTALL
    )
    if not args_match:
        return {}

    # Extract parameter descriptions
    args_text = args_match.group(1).strip()
    param_descriptions = {}

    # Process line by line to avoid capturing descriptions of other parameters
    lines = args_text.split("\n")
    current_param = None
    current_desc = []

    for line in lines:
        line = line.strip()
        # Find a new parameter
        param_match = re.match(r"(\w+)\s*\((.*?)\):\s*(.+)", line)

        if param_match:
            # Save description of previous parameter if exists
            if current_param:
                param_descriptions[current_param] = " ".join(current_desc).strip()

            # Start new parameter
            current_param = param_match.group(1)
            current_desc = [param_match.group(3)]
        elif current_param and line:
            # Continue description of current parameter
            current_desc.append(line)

    # Save last description
    if current_param:
        param_descriptions[current_param] = " ".join(current_desc).strip()

    return param_descriptions


def generate_json_schema(cls: type) -> Dict:
    """
    Generates a JSON schema for a class based on its characteristics.

    Args:
        cls: The class to generate the schema for

    Returns:
        JSON schema for the class
    """
    name = cls.name if hasattr(cls, "name") else cls.__name__
    description = cls.__docstring__ if hasattr(cls, "__docstring__") else cls.__doc__
    clean_description = clean_docstring(description)
    param_descriptions = parse_docstring_args(description)
    annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}

    properties = {}
    required = []

    for param, type_hint in annotations.items():

        if param == "return":
            continue

        prop_schema = {"type": "string"}  # Default as string

        # Check if enum is defined
        if hasattr(type_hint, "__args__") and type_hint.__origin__ is Literal:
            prop_schema["enum"] = list(type_hint.__args__)

        # Add parameter description if available
        if param in param_descriptions:
            prop_schema["description"] = param_descriptions[param]

        # Mark as required
        if not get_origin(type_hint) is Union:
            required.append(param)

        properties[param] = prop_schema

    json_schema = {
        "name": name,
        "description": clean_description or f"Function for {name}",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    return json_schema

# TODO: needs improvement to write encoded json
def get_react_tools_prompt_format(tool_schemas):
    template = Template("""
    You are a function calling AI model. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:


    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endif %}

    For each function call return a encoded json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
    """)    
    react_tools = template.render(tools=tool_schemas)
    return react_tools

