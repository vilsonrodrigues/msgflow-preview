from pathlib import Path
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Literal, 
    Optional, 
    Union
)

import msgspec
from jinja2 import Template

from msgflow.generation.plan.react import ReAct
from msgflow.logger import logger
from msgflow.models.gateway import ModelGateway
from msgflow.message import Message
from msgflow.models.types import ChatCompletionModel
from msgflow.nn.modules.module import Module
from msgflow.nn.modules.tool import ToolLibrary
from msgflow.nn.parameter import Parameter
from msgflow.utils.chat import (
    chatml_to_steps_format, 
    download_file, 
    get_filename, 
    get_react_tools_prompt_format
)
from msgflow.utils.encode import encode_data_to_base64
from msgflow.utils.encode import get_mime_type
from msgflow.utils.validation import is_base64, is_subclass_of
from msgflow.telemetry.span import trace_agent_prepare_model_execution


# Context Manager function that will manage the processing of assembling the context
# new features: context_cache fixed message in the model, in the retrieval
# initial_assist_msg or prefix_agent_msg this will be a param (prefilling)
# it is possible to continue generating a model. Just resend to it what it
# wrote and then it will continue from there

# from schema to scheme

# if not all(isinstance(module, Module) for module in modules_to_send):

# the system can change the response to the stream if x condition is met. nein

# add time/date to the system prompt (this can be bad if you use prompt cache)

class PromptSpec:
    SYSTEM_MESSAGE = "Who are you"
    INSTRUCTIONS = "How you should do"
    EXAMPLES = "Samples of what to do"
    EXPECTED_OUTPUT = "Describes what the response should be like"
    SYSTEM_PROMPT_TEMPLATE = "A jinja template to format the system prompt"


SYSTEM_PROMPT_TEMPLATE =  """
{% if system_message or instructions or expected_output or examples %}
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
</developer_note>
{% endif %}
"""


class Agent(Module):
    r"""Agent is a Module type that uses language models to solve tasks.

    An Agent can perform actions in an environment using function calls.
    For an Agent, a function is any callable object.

    An Agent can handle multimodal inputs and outputs.

    Args:
        name: Agent name in snake case format.
        model: ChatCompletation Model client.
        system_message: The Agent behaviour.
        instructions: What the Agent should do.
        expected_output: What the response should be like.
        stream: If the response is transmitted on-fly.initial_assist_msg
        task_template: Template to task.
        task_inputs: Fields of the Message object that will be the input to the task.
        task_multimodal_inputs: Fields of the Message object that will be the multimodal
            input to the task.
        context_inputs: Fields of the Message object that will be the context to the task.
        structured_output: A msgspec.Struct class to specify the structured output.
        response_mode: What the response should be. Has five options:
            * `plain_response` (default): Returns the final agent response
            * `steps`: Returns a structured model state. Containing user input, function calls, and the final response.
            * `response`: Write on `response` field in Message object.
            * `context`: Write on `context` field in Message object. It`s insert how `context.agent_name`.
            * `outputs`: Write on `outputs` field in Message object. It`s insert how `outputs.agent_name`.
        tools:
            ...
        response_template:
            ...
        prefilling:
            ...
        context_cache:
            ...
            ...
        fixed_messages:
            ...
        description:
            The Agent description (docstring). It's useful when using an agent-as-a-function.
        _annotations
            Define the input and output annotations to use the agente-as-a-function.
            Default is: `{"message": str, "return": str}`

    !!! example
        ``` python
        import msgflow.nn as nn
        model = sonnet
        writter_agent = nn.Agent(name="writter_agent", model=model)
        response = writter_agent("What's Deep Learning?")
        print(response)
            ```
    """

    _supported_outputs: List[str] = [
        "structured",
        "text_generation",
        "audio_generation",
        "audio_text_generation",
    ]

    def __init__(
        self,
        name: str,
        model: Union[ChatCompletionModel, ModelGateway],
        *,
        system_message: Optional[str] = None,
        instructions: Optional[str] = None,
        expected_output: Optional[str] = None,
        examples: Optional[str] = None,
        stream: Optional[bool] = False,
        input_guardrail: Optional[Callable] = None,
        output_guardrail: Optional[Callable] = None,
        task_inputs: Optional[Union[str, Dict[str, str]]] = None,
        task_template: Optional[str] = None,
        task_multimodal_inputs: Optional[Dict[str, List[str]]] = None,
        context_inputs: Optional[Union[str, List[str]]] = None,
        model_preference: Optional[str] = None,
        prefilling: Optional[str] = None,
        context_cache: Optional[str] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        response_mode: Optional[str] = "plain_response",
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        response_template: Optional[str] = None,
        task_messages: Optional[str] = None,                
        # task_messages_mode: Literal["relevant", "recent", "full"] = "relevant",
        fixed_messages: Optional[List[Dict[str, Any]]] = None,
        #signature: Optional[str] = None,
        #verbose: Optional[bool] = False,
        description: Optional[str] = None,
        system_prompt_template: Optional[str] = SYSTEM_PROMPT_TEMPLATE,
        _annotations: Optional[Dict[str, type]] = None,
    ):
        super().__init__()

        if stream and response_template:
            raise ValueError("`response_template` is not `stream=True` compatible")
        if stream and is_subclass_of(generation_schema, ReAct): # TODO nenhum deles deve ser
            raise ValueError(
                "`generation_schema=ReAct` is not `stream=True` compatible"
            )
        if stream and output_guardrail:
            raise ValueError("`output_guardrail` is not `stream=True` compatible")

        self.set_name(name)
        self.set_description(description)
        self._set_annotations(_annotations or {"message": str, "return": str})
        self._set_context_cache(context_cache)
        self._set_context_inputs(context_inputs)
        self._set_examples(examples)
        self._set_expected_output(expected_output)
        self._set_fixed_messages(fixed_messages)
        self._set_generation_schema(generation_schema)
        self._set_input_guardrail(input_guardrail)
        self._set_output_guardrail(output_guardrail)
        self._set_instructions(instructions)
        self._set_task_messages(task_messages)
        self._set_model(model)        
        self._set_model_preference(model_preference)
        self._set_prefilling(prefilling)
        self._set_system_message(system_message)
        self._set_system_prompt_template(system_prompt_template)
        self._set_response_mode(response_mode)
        self._set_stream(stream)
        self._set_response_template(response_template)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_task_inputs(task_inputs)
        self._set_task_template(task_template)
        self._set_tool_choice(tool_choice)        
        self._set_tools(tools)

    def forward(self, message: Union[str, Dict[str, Any], Message]):
        model_preference = self.get_model_preference(message)
        model_state = self._prepare_task(message)
        model_response = self._execute_model(model_state, self.prefilling.data, model_preference)
        response = self._process_model_response(model_response, model_state, message, model_preference)
        return response

    def _execute_model(self, model_state, prefilling=None, model_preference=None):
        model_execution_params = self._prepare_model_execution(model_state, prefilling, model_preference)
        if self.attr_is_valid(self.input_guardrail):
            self._execute_input_guardrail(model_execution_params)
        model_response = self.model.data(**model_execution_params)
        return model_response

    @trace_agent_prepare_model_execution
    def _prepare_model_execution(self, model_state, prefilling=None, model_preference=None):
        agent_state = []

        if self.fixed_messages.data:
            agent_state.extend(self.fixed_messages.data)

        agent_state.extend(model_state)

        system_prompt = self._get_system_prompt()

        tool_schemas = self.tool_library.get_tool_json_schemas()
        if not tool_schemas:
            tool_schemas = None

        if is_subclass_of(self.generation_schema, ReAct) and tool_schemas:
            react_tools = get_react_tools_prompt_format(tool_schemas)
            if system_prompt: # TODO: invert
                react_tools += f"\n\n {system_prompt}"
            else:
                system_prompt = react_tools
            # Disable tool_schemas to react controlflow preference
            tool_schemas = None

        model_execution_params = {
            "messages": agent_state,
            "system_prompt": system_prompt or None,
            "prefilling": prefilling,
            "stream": self.stream.data,
            "tool_schemas": tool_schemas,
            "tool_choice": self.tool_choice.data,
            "generation_schema": self.generation_schema.data,
        }

        if model_preference:
            model_execution_params["model_preference"] = model_preference

        return model_execution_params

    def _prepare_input_guardrail_execution(self, model_execution_params):
        model_state = model_execution_params.get("model_state")
        last_message = model_state[-1]
        if isinstance(last_message.get("content"), list):
            if last_message.get("content")[0]["type"] == "image_url":
                data = [last_message]
            else: # audio, file
                data = last_message.get("content")[-1] # text input
        else:
            data = last_message.get("content")
        guardrail_params = {"data": data}
        return guardrail_params

    def _process_model_response(self, model_response, model_state, message, model_preference):
        if model_response.response_type == "tool_call":
            model_response, model_state = (
                self._process_tool_call_response(model_response, model_state, model_preference)
            )
        elif is_subclass_of(self.generation_schema, ReAct):
            model_response, model_state = self._process_react_response(
                model_response, model_state, model_preference
            )
        
        raw_response = self._extract_raw_response(model_response)

        response_type = model_response.response_type

        if model_response.response_type in self._supported_outputs:
            response = self._prepare_response(
                raw_response, 
                model_response.response_type,
                model_state, 
                message
            )
            return response
        else:
            raise ValueError(f"Unsupported `response_type={response_type}`")

    def _process_react_response(self, model_response, model_state, model_preference=None):
        while True:            
            raw_response = self._extract_raw_response(model_response)

            if raw_response.get("current_step"):
                actions = raw_response["current_step"]["actions"]
                tool_callings = [
                    (act["id"], act["name"], act["arguments"]) for act in actions
                ]
                tool_responses = self._process_tool_call(tool_callings)

                for act in actions:
                    act["result"] = tool_responses[[act["id"]]]

                if model_state[-1]["role"] == "assistant":
                    last_react_msg = model_state[-1]["content"]
                    react_state = msgspec.json.decode(last_react_msg)
                    react_state.append(raw_response)
                    react_state_encoded = msgspec.json.encode(react_state)
                    model_state[-1] = react_state_encoded
                else:
                    react_state = []
                    react_state.append(raw_response)
                    react_state_encoded = msgspec.json.encode(react_state)
                    model_state.append(
                        [{"role": "assistant", "content": react_state_encoded}]
                    )

            elif raw_response.get("final_answer"):
                return model_response, model_state

            model_response = self._execute_model(model_state, model_preference=model_preference)

    def _process_tool_call_response(self, model_response, model_state, model_preference=None):
        """
        Mensagens: [{'role': 'assistant', 'tool_calls': [{'id': 'call_1YLHAVwHwDPjEBuMpWQfSktO',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_1YLHAVwHwDPjEBuMpWQfSktO',
        'content': '2024-10-15'}]
        """
        while True:
            if model_response.response_type == "tool_call":
                raw_response = self._extract_raw_response(model_response)
                tool_callings = raw_response.get_calls()
                tool_responses = self._process_tool_call(tool_callings)
                raw_response.insert_results(tool_responses)
                tool_responses_message = raw_response.get_messages()
                model_state.extend(tool_responses_message)
            else:
                return model_response, model_state

            model_response = self._execute_model(model_state, model_preference=model_preference)

    def _process_tool_call(self, tool_callings):        
        tool_responses = self.tool_library(tool_callings)
        return tool_responses

    def _prepare_response(self, raw_response, response_type, model_state, message):
        if response_type in ["text_generation", "structured"]:
            if self.attr_is_valid(self.output_guardrail):
                self._execute_output_guardrail(raw_response)        
            if self.response_template.data:
                response = self._format_response_template(raw_response)
        else:
            response = raw_response

        return self._define_response_mode(response, model_state, message)

    def _prepare_output_guardrail_execution(self, model_response):
        if isinstance(model_response, str):
            data = model_response
        else:
            data = str(model_response)
        guardrail_params = {"data": data}
        return guardrail_params

    def _define_response_mode(self, response, model_state, message):
        if self.response_mode.data == "plain_response":
            return response
        elif self.response_mode.data == "steps":
            return self._apply_steps_format(model_state, response)
        elif isinstance(message, Message):
            if self.response_mode.data.startswith(("context", "outputs", "response")):
                message.set(f"{self.response_mode.data}.{self.name.data}", response)
            return message
        else:
            raise ValueError(
                "For `response_mode` other than `plain_response` and "
                "`steps` the message object must be of type Message"
            )

    def _apply_steps_format(self, model_state, response):
        steps_response = chatml_to_steps_format(model_state, response)
        return steps_response

    def _prepare_task(
        self, message: Union[str, Dict[str, Any], Message]
    ) -> List[Dict[str, Any]]:        
        """Prepare model input in ChatML format"""
        task_messages = None
        
        if isinstance(message, (str, dict)):
            content = self._process_str_dict_task(message)
        elif isinstance(message, Message):
            content = self._process_message_task(message)
            task_messages = self._get_task_messages(message)
        else:
            raise ValueError("Unsupported message type")
        
        if content is None and task_messages is None:
            raise ValueError("No data was detected to make the model input")

        if content is not None:
            chat_content = [{"role": "user", "content": content}]
            if task_messages is None:
                return chat_content
            else:
                task_messages.extend(chat_content)
                return task_messages
        else:
            return task_messages

    def _process_str_dict_task(self, message: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.task_template.data:
            content = self._format_task_template(message)
            return content
        else:
            if isinstance(message, dict):
                raise AttributeError("message is a dict that requires a `task_template`")
            return message

    def _process_message_task(self, message: Message) -> Optional[Union[str, List[Dict[str, Any]]]]:
        content = ""

        # Process context
        context_content = self._context_manager(message)

        if context_content:
            content += context_content

        # Process text content
        if self.task_inputs.data:
            text_content = self._process_inputs(message)
            if self.task_template.data:
                text_content = self._format_task_template(text_content)
            content += f"<task>\n{text_content}\n</task>\n"
        # It's possible to use `task_template` as the default task message
        # if no `task_inputs` is selected. This can be useful for multimodal
        # models that require a text message to be sent along with the data
        elif self.task_template.data:
            content += f"<task>\n{self.task_template.data}\n</task>\n"

        # Remove whitespace
        content = content.strip()

        # Process multimodal content        
        if self.task_multimodal_inputs.data:
            multimodal_content = self._process_multimodal_inputs(message)
            if multimodal_content:
                multimodal_content.append({"type": "text", "text": content})
                return multimodal_content

        return content

    def _process_inputs(self, message: Message) -> Union[str, Dict[str, Any]]:
        # TODO allow other fields besides outputs?
        content = None
        
        if self.task_inputs == "outputs":  # Consume all values in outputs
            content = "\n\n".join(str(v) for v in message.get("outputs").values())
        elif isinstance(self.task_inputs.data, str):
            content = self._get_content_from_message(self.task_inputs.data, message)
        elif isinstance(self.task_inputs.data, dict):
            text_inputs = {}
            for k, v in self.task_inputs.data.items():
                text_inputs[k] = self._get_content_from_message(v, message)
            content = text_inputs
            
        return content
    
    def _context_manager(self, message: Message): # TODO support to list, dict, str
        """ Manager agent context """
        content = ""
        
        if self.context_cache.data:
            content += f"{self.context_cache.data}\n\n"
            
        if isinstance(self.context_inputs.data, str) and self.context_inputs.data == "context":
            msg_context = "\n\n".join(str(v) for v in message.get("context").values())
        elif isinstance(self.context_inputs.data, list): # ["context.1", "context.2"]
            context_values = []
            for path in self.context_inputs.data:

                context_value = self._get_content_from_message(path, message)
                if context_value is not None:
                    context_values.append(context_value)

            msg_context = " ".join(context_values)
        else:
            msg_context = None
        
        if msg_context:
            content += f"{msg_context}\n\n"                        

        if content:
            context_content = f"<context>\n{content}\n</context>\n"
            return context_content
        else:
            return None

    def _process_multimodal_inputs_old(self, message: Message) -> List[Dict[str, Any]]:
        # TODO: suporte para consumir todas as entradas de images or outro
        content = []
        
        if isinstance(self.task_multimodal_inputs.data, dict):
            for image_path in self.task_multimodal_inputs.data.get("image", []):
                image_data = self._get_content_from_message(image_path, message)
                if image_data:
                    if not image_data.startswith("http") and not is_base64(image_data):
                        base64_image = encode_local_file_in_base64(image_data)
                        image_data = f"data:image/jpeg;base64,{base64_image}"
                    content.append({"type": "image_url", "image_url": {"url": image_data}})

            for audio_path in self.task_multimodal_inputs.data.get("audio", []):
                audio_data = self._get_content_from_message(audio_path, message)
                if audio_data:
                    audio_format = Path(audio_data).suffix

                    if not audio_data.startswith("http") and not is_base64(audio_data):
                        base64_audio = encode_local_file_in_base64(audio_data)
                    elif audio_data.startswith("http"):
                        base64_audio = encode_base64_from_url(audio_data)
                    else:
                        base64_audio = audio_data

                    content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": base64_audio, "format": audio_format},
                        }
                    )

            for file_path in self.task_multimodal_inputs.data.get("file", []):
                file_data = self._get_content_from_message(file_path, message)                
                if file_data:
                    filename = get_filename(file_data)
                    if file_data.startswith("http"):
                        file_data = download_file(file_data)                        
                    if file_data is not is_base64(file_data):
                        base64_pdf = encode_local_file_in_base64(file_data)
                    file_data = f"data:application/pdf;base64,{base64_pdf}"                    
                    content.append(
                        {"type": "file", "file": {"filename": filename, "file_data": file_data}}
                    )

        return content

    def _prepare_data_uri(self, source: str, force_encode: bool = False) -> str:
        """
        Prepares a data string (URL or Data URI base64).
        If force_encode=True, always tries to download and encode URL.
        Otherwise, keeps the URL if it is HTTP and not base64.
        Returns None in case of encoding/download error.
        """
        if not source:
            return None

        if is_base64(source):
            # If it is already base64, assume it is ready (no prefix)
            # Prefix will be added by formatter if needed
            return source

        is_url = source.startswith("http")

        if is_url and not force_encode:
             # Keep the URL as is if you don't force the encoding
             return source

        # Need to encode (either local or force_encode=True for URL)
        try:
            return encode_data_to_base64(source)
        except Exception as e:
            logger.error(f"Failed to encode source {source}: {e}")
            return None

    def _format_image_input(self, image_source: str) -> Dict[str, Any]:
        """Formats the image input for the model"""
        base64_image = self._prepare_data_uri(image_source, force_encode=True)

        if not base64_image:
            return None

        mime_type = get_mime_type(image_source) # Try to guess from the original source
        if not mime_type.startswith("image/"): mime_type = "image/jpeg" # Fallback        
        image_data_url = f"data:{mime_type};base64,{base64_image}"
        
        return {"type": "image_url", "image_url": {"url": image_data_url}}

    def _format_audio_input(self, audio_source: str) -> Dict[str, Any]:
        """Formats the audio input for the model"""
        base64_audio = self._prepare_data_uri(audio_source, force_encode=True)

        if not base64_audio:
            return None        

        audio_format_suffix = Path(audio_source).suffix.lstrip(".")
        mime_type = get_mime_type(audio_source)
        if not mime_type.startswith("audio/"):
             # If MIME type is not audio, use suffix or fallback
             audio_format_for_uri = audio_format_suffix if audio_format_suffix else "mpeg" # fallback
             mime_type = f"audio/{audio_format_for_uri}"

        # Use suffix like 'format' if available, otherwise extract from mime type
        format_key = audio_format_suffix if audio_format_suffix else mime_type.split("/")[-1]
        return {
            "type": "input_audio",
            "input_audio": {"data": base64_audio, "format": format_key},
        }

    def _format_file_input(self, file_source: str) -> Dict[str, Any]:
        """Formats the file input for the model"""
        base64_file = self._prepare_data_uri(file_source, force_encode=True)

        if not base64_file:
            return None

        filename = get_filename(file_source)
        mime_type = get_mime_type(file_source)

        if mime_type == "application/octet-stream" and filename.lower().endswith(".pdf"):
            mime_type = "application/pdf"

        file_data_uri = f"data:{mime_type};base64,{base64_file}"

        return {
            "type": "file",
            "file": {"filename": filename, "file_data": file_data_uri}
        }

    def _process_multimodal_inputs(self, message: Message) -> List[Dict[str, Any]]:
        """
        Processes multimodal inputs (image, audio, file) from the configuration
        and the Message object, returning a list of dictionaries formatted for the model.
        """
        content = []
        multimodal_config = self.task_multimodal_inputs.data

        formatters = {
            "image": self._format_image_input,
            "audio": self._format_audio_input,
            "file": self._format_file_input,
        }

        for media_type, formatter in formatters.items():
            
            path_keys = multimodal_config.get(media_type, [])
            if not isinstance(path_keys, list):
                 logger.warning("Warning: Expected list for multimodal config key "
                                f"`{media_type}`, got `{type(path_keys)}`")
                 continue # Skip this media type if the config is badly formatted

            for path_key in path_keys:
                media_source = self._get_content_from_message(path_key, message)
                if media_source:
                    formatted_input = formatter(media_source)
                    if formatted_input:
                        content.append(formatted_input)
                else:
                    logger.debug(f"No valid datat to `path_key={path_key}`")

        return content

    def _get_task_messages(self, message: Message) -> Optional[List[Dict[str, Any]]]:
        """Returns a message history (ChatML format) from message"""        
        messages_history = None
        if self.task_messages.data:
            messages_history = self._get_content_from_message(self.task_messages.data, message)        
        return messages_history

    def _set_context_inputs(self, context_inputs: Optional[Union[str, List[str]]] = None):
        if isinstance(context_inputs, (str, list)) or context_inputs is None:
            if isinstance(context_inputs, str) and context_inputs == "":
                raise ValueError("`context_inputs` requires a string not empty" 
                                 f"given `{context_inputs}`")
            if isinstance(context_inputs, list) and not context_inputs:
                raise ValueError("`context_inputs` requires a list not empty"
                                 f"given `{context_inputs}`")
            self.register_buffer("context_inputs", context_inputs)
        else:
            raise TypeError("`context_inputs` requires a string, list or None"
                            f"given `{type(context_inputs)}`")

    def _set_context_cache(self, context_cache: Optional[str] = None):
        if isinstance(context_cache, str) or context_cache is None:
            self.register_buffer("context_cache", context_cache)
        else:
            raise TypeError("`context_cache` requires a string or None"
                            f"given `{type(context_cache)}`")

    def _set_prefilling(self, prefilling: Optional[str] = None):
        if isinstance(prefilling, str) or prefilling is None:
            self.register_buffer("prefilling", prefilling)
        else:
            raise TypeError("`prefilling` requires a string or None"
                            f"given `{type(prefilling)}`")        

    def _set_response_mode(self, response_mode: str):
        if isinstance(response_mode, str):
            if (
                response_mode in ["plain_response", "steps","response"] 
                or 
                response_mode.startswith(("context", "outputs"))
            ):
                self.register_buffer("response_mode", response_mode)            
            else:
                raise ValueError(
                    f"`response_mode={response_mode}` is not supported "
                    "only `plain_response`, `steps`, `context`, `outputs` "
                    "and `response`"
                )
        else:
            raise TypeError("`response_mode` requires a string "
                            f"given `{type(response_mode)}`")

    # def _set_chat_history(self, chat_history: Union[ChatHistory, MultiChatHistory]):
    #    super().__setattr__("chat_history", chat_history)

    def _set_tools(self, tools: Optional[List[Callable]] = None):
        if (
            (isinstance(tools, list) and all(callable(obj) for obj in tools)) 
            or 
            tools is None
        ):
            self.tool_library = ToolLibrary(self.get_module_name(), tools or [])
        else:
            raise TypeError("`tools` need be a list of callables or None"
                            f"given `{type(tools)}`")        

    def _set_fixed_messages(self, fixed_messages: Optional[List[Dict[str, Any]]] = None):
        if (
            (isinstance(fixed_messages, list) and all(dict(obj) for obj in fixed_messages)) 
            or 
            fixed_messages is None
        ):
            self.register_buffer("fixed_messages", fixed_messages)
        else:
            raise TypeError("`fixed_messages` need be a list of dict or None"
                            f"given `{type(fixed_messages)}`")

    def _set_generation_schema(self, generation_schema: Optional[msgspec.Struct] = None):
        if is_subclass_of(generation_schema, msgspec.Struct) or generation_schema is None:
            self.register_buffer("generation_schema", generation_schema)
        else:
            raise TypeError("`generation_schema` need be a `msgspec.Struct` or None"
                            f"given `{type(generation_schema)}`")

    def _set_model(self, model: Union[ChatCompletionModel, ModelGateway]):
        if model.model_type == "chat_completion":
            self.register_buffer("model", model)
        else:
            raise TypeError(f"`model` need be a `chat completion` model, given `{type(model)}`")

    def _set_tool_choice(self, tool_choice: Optional[str] = None):
        if isinstance(tool_choice, str) or tool_choice is None:
            if isinstance(tool_choice, str):
                if tool_choice not in ["auto", "required"]:
                    tool_choice = {"type": "function", "function": {"name": tool_choice}}
            self.register_buffer("tool_choice", tool_choice)
        else:
            raise TypeError("`tool_choice` need be a str or None "
                            f"given `{type(tool_choice)}`")            

    def _set_system_message(self, system_message: Optional[str] = None):
        if isinstance(system_message, str) or system_message is None:
            self.system_message = Parameter(system_message, PromptSpec.SYSTEM_MESSAGE)
        else:
            raise TypeError("`system_message` requires a string or None "
                            f"given `{type(system_message)}`")

    def _set_instructions(self, instructions: Optional[str] = None):
        if isinstance(instructions, str) or instructions is None:
            self.instructions = Parameter(instructions, PromptSpec.INSTRUCTIONS)
        else:
            raise TypeError("`instructions` requires a string or None "
                             f"given `{type(instructions)}`")

    def _set_expected_output(self, expected_output: Optional[str] = None):
        if isinstance(expected_output, str) or expected_output is None:
            self.expected_output = Parameter(
                expected_output, PromptSpec.EXPECTED_OUTPUT
            )
        else:
            raise TypeError("`expected_output` requires a string or None "
                            f"given `{type(expected_output)}`")

    def _set_examples(self, examples: Optional[str] = None):
        if isinstance(examples, str) or examples is None:
            self.examples = Parameter(examples, PromptSpec.EXAMPLES)
        else:
            raise TypeError("`examples` requires a string or None "
                            f"given `{type(examples)}`")

    def _set_task_messages(self, task_messages: Optional[str] = None):
        if isinstance(task_messages, str) or task_messages is None:
            self.register_buffer("task_messages", task_messages)
        else:
            raise TypeError("`task_messages` requires a string or None "
                            f"given `{type(task_messages)}`")

    def _set_system_prompt_template(self, system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE):
        if isinstance(system_prompt_template, str) or system_prompt_template is None:
            self.examples = Parameter(system_prompt_template, PromptSpec.SYSTEM_PROMPT_TEMPLATE)
        else:
            raise TypeError("`system_prompt_template` requires a string given "
                            f"`{type(system_prompt_template)}`")
        
    def _get_system_prompt(self):
        """
        Render the system prompt using the Jinja template.
        Returns an empty string if no segments are provided.
        """
        template = Template(self.system_prompt_template.data)
        system_prompt = template.render(
            system_message=self.system_message.data,
            instructions=self.instructions.data,
            expected_output=self.expected_output.data,
            examples=self.examples.data,
        )
        return system_prompt
