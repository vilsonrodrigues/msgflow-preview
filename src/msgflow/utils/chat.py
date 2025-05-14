import copy
import os
import re
import requests
import tempfile
from uuid import uuid4
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    List,
    Optional,
    Union,
    Tuple,
    get_origin,
)
from jinja2 import Template
from urllib.parse import urlparse
from msgflow.logger import logger
from msgflow.utils.inspect import get_mime_type
from msgflow.utils.xml import apply_xml_tags


class PromptSpec:
    SYSTEM_MESSAGE = "Who are you"
    INSTRUCTIONS = "How you should do"
    EXAMPLES = "Samples of what to do"
    EXPECTED_OUTPUT = "Describes what the response should be like"
    SYSTEM_PROMPT_TEMPLATE = "A jinja template to format the system prompt"
    #TASK_TEMPLATE = ""


SYSTEM_PROMPT_TEMPLATE =  """
{% if system_message or instructions or expected_output or examples or system_extra_message %}
<developer_note>
{% if system_message %}{{ system_message }}
{% endif %}
{% if instructions %}<instructions>
{{ instructions }}
</instructions>
{% endif %}
{% if expected_output %}<expected_output>
{{ expected_output }}
</expected_output>
{% endif %}
{% if examples %}<examples>
{{ examples }}
</examples>
{% endif %}
{% if system_extra_message %}
{{ system_extra_message }}
{% endif %}
</developer_note>
{% endif %}
"""

XML_TO_DICT_TEMPLATE =  """
{% if instructions %}{{ instructions }}{% endif %}

You SHOULD write your response in a structured manner using XML tags.
DO NOT write XML headers or add extra messages beyond the XML response.
You should then generate an XML specifying the dtype.
The available data types are: str (default if not specified), int, float, bool, dict and list.

Example of how you can write your response in XML:

<user_profile dtype="dict">
    <id dtype="int">1024</id>
    <username dtype="str">johndoe</username>
    <is_active dtype="bool">true</is_active>
    <account_balance dtype="float">2.75</account_balance>

    <preferences dtype="dict">
        <newsletter_subscribed dtype="bool">false</newsletter_subscribed>
        <theme>dark</theme>
    </preferences>

    <roles dtype="list">
        <role>admin</role>
        <role>editor</role>
    </roles>

    <login_history dtype="list">
        <login_event dtype="dict">
            <ip_address dtype="str">192.168.1.100</ip_address>
            <successful dtype="bool">true</successful>
        </login_event>
        <login_event dtype="dict">
            <ip_address dtype="str">192.168.1.0</ip_address>
            <successful dtype="bool">false</successful>
        </login_event>
    </login_history>
</user_profile>

{% if json_schema %}
Here is a JSON-schema that you SHOULD use to guide you in generating your response using XML tags:
{{ json_schema }}
{% endif %}
"""


def format_examples(examples: List[Union[Tuple[str, str], Tuple[str, str, str]]]) -> str:
    """
    Formats a list of examples into XML-style string format.
    
    Each example in the list should be a tuple containing input and output strings,
    with an optional title as the third element. The function generates sequential IDs
    for each example starting from 1.
    
    Args:
        examples: A list of tuples where each tuple contains:
            - Input string (required)
            - Output string (required)
            - Title string (optional)
    
    Returns:
        A formatted XML-style string containing all examples.
    
    Example:
        >>> examples = [
        ...     ("What is your name?", "My name is GPT-3.5.", "Introduction"),
        ...     ("What day is today?", "Today is Tuesday."),
        ... ]
        >>> print(format_examples(examples))
        <example id=1 title="Introduction">
        <input>What is your name?</input>
        <output>My name is GPT-3.5.</output>
        </example>
        
        <example id=2>
        <input>What day is today?</input>
        <output>Today is Tuesday.</output>
        </example>
    """
    result = []
    
    for i, example in enumerate(examples, start=1):
        if len(example) == 3:
            input_text, output_text, title = example
            result.append(f'<example id={i} title="{title}">')
        else:
            input_text, output_text = example
            result.append(f"<example id={i}>")
        
        result.append(apply_xml_tags("input", input_text))
        result.append(apply_xml_tags("output", output_text))
        result.append("</example>\n")
    
    return "\n".join(result)

def format_available_workers(workers: List[Dict[str, str]], title: Optional[str] = None) -> str:
    """
    Generates an XML structure for a list of workers, mapping module names to their descriptions.

    Args:
        workers: 
            A list of dictionaries, each containing 'name' and 'description' keys.
        title: 
            A title in workers.

    Returns:
        str: A string containing the XML structure with module names and descriptions.

    Example:
        workers = [
            {"name": "Authentication", "description": "Handles user login and session management."},
        ]
        xml = format_available_workers(workers)
        print(xml)
        <workers>
        <worker id=1 name="Authentication">
        <description>
        Handles user login and session management.
        </description>
        </worker>
        </workers>
    """
    id = output_id = "workers"
    sub_id = "worker"
    modules_content = ""

    for i, module in enumerate(workers, start=1):
        description_xml = apply_xml_tags("description", module["description"])
        module_xml = apply_xml_tags(
            f'{sub_id} id={i} name="{module["name"]}"', 
            description_xml, 
            output_id=sub_id
        )
        modules_content += f"{module_xml}\n"

    if title:
        id += f'title="{title}"'

    return apply_xml_tags(id, modules_content.strip(), output_id)


def adapt_struct_schema_to_json_schema(
    original_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a Msgspec.Struct in Json Schema ChatCompletion-like"""
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
                    "results": None, # To be updated when the answer is found
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
        steps.append({"assistant": response})

    return steps

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


def generate_json_schema(cls: type) -> Dict[str, Any]:
    """
    Generates a JSON schema for a class based on its characteristics.

    Args:
        cls: The class to generate the schema for

    Returns:
        JSON schema for the class
    """
    name = cls.get_module_name()
    description = cls.get_module_description()
    clean_description = clean_docstring(description)
    param_descriptions = parse_docstring_args(description)
    annotations = cls.get_module_annotations()

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
            'additionalProperties': False,
        },
        "strict": True,
    }

    return json_schema

def generate_tool_json_schema(cls: type) -> Dict[str, Any]:
    tool = generate_json_schema(cls)
    tool_json_schema = {
        "type": "function",
        "function": tool
    }
    return tool_json_schema

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

def get_filename(data_path: str) -> str:
    if data_path.startswith(("http://", "https://", "ftp://")):
        parsed_url = urlparse(data_path)
        filename = os.path.basename(parsed_url.path)
    # Local file
    else:
        filename = os.path.basename(data_path)    
    return filename

def download_file(url: str) -> str:
    """ Download a webfile and returns the path """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = f"downloaded_file_{str(uuid4())}"

        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return file_path

    except requests.exceptions.RequestException as e:
        logger.error(str(e))
        return None

def adapt_messages_for_vllm_audio(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adapts a list of messages from ChatML format, converting audio parts of type 
    'input_audio' (OpenAI style) to type 'audio_url' with Data URI (vLLM style).

    Args:
        messages: The original list of messages.

    Returns:
        A new list of messages with the adapted audio parts.
        The original list is not modified.
    """
    adapted_messages = copy.deepcopy(messages)

    for message in adapted_messages:
        content = message.get("content")

        # Checks if the content is a list (indicating multimodality)
        if isinstance(content, list):
            processed_content = []
            for i, part in enumerate(content):
                # Check if the part is of type 'input_audio'
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    input_audio_data = part.get("input_audio")

                    # Check if internal data exists
                    if isinstance(input_audio_data, dict):
                        base64_data = input_audio_data.get("data")
                        audio_format = input_audio_data.get("format")

                        # If you have the base64 data and format, convert
                        if base64_data and isinstance(base64_data, str) and audio_format:
                            mime_type = get_mime_type(audio_format)
                            data_uri = f"data:{mime_type};base64,{base64_data}"

                            # Create the new structure of the audio part
                            vllm_audio_part = {
                                "type": "audio_url",
                                "audio_url": {"url": data_uri}
                            }
                            processed_content.append(vllm_audio_part)
                        else:
                            logger.warning("Warning: Skipping malformed 'input_audio' part "
                                           "at index {i}: {part}")
                            processed_content.append(part)
                    else:
                        # Keep the original part if 'input_audio' is not a dict
                        logger.warnning("Warning: Skipping malformed 'input_audio' part "
                              f"(not a dict) at index {i}: {part}")
                        processed_content.append(part)

                else:
                    # Keep other parts (text, image, etc.) as is
                    processed_content.append(part)

            # Update the message content with the processed list
            message["content"] = processed_content
        # If the content is not a list (e.g. plain text), do nothing
    return adapted_messages