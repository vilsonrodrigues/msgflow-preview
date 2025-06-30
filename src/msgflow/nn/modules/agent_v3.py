from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Tuple
)

import msgspec

from msgflow.dotdict import dotdict
from msgflow.generation.reasoning.react import ReAct
from msgflow.generation.signature import (
    Signature,
    SIGNATURE_SYSTEM_MESSAGES,
    get_examples_from_signature,
    get_expected_output_from_signature,
    get_task_template_from_signature,
)
from msgflow.generation.templates import (
    PromptSpec,
    SIGNATURE_DEFAULT_SYSTEM_MESSAGE,    
    SYSTEM_PROMPT_TEMPLATE,
    XML_TO_DICT_TEMPLATE,
)
from msgflow.logger import logger
from msgflow.message import Message
from msgflow.models.gateway import ModelGateway
from msgflow.models.types import ChatCompletionModel
from msgflow.models.response import ModelResponse, ModelStreamResponse
from msgflow.nn.modules.module import Module
from msgflow.nn.modules.tool import ToolLibrary
from msgflow.nn.parameter import Parameter
from msgflow.utils.chat import (
    adapt_struct_schema_to_json_schema,
    chatml_to_steps_format,
    format_examples,
    get_filename, 
    get_react_tools_prompt_format
)
from msgflow.utils.inspect import get_mime_type
from msgflow.utils.msgspec import StructFactory
from msgflow.utils.validation import is_subclass_of
from msgflow.utils.xml import apply_xml_tags
from msgflow.telemetry.span import trace_agent_prepare_model_execution


# it is possible to continue generating a model. Just resend to it what it
# wrote and then it will continue from there
# the system can change the response to the stream if x condition is met. nein


class Agent(Module):

    _supported_outputs: List[str] = [
        "reasoning_structured",
        "reasoning_text_generation",
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
        task_multimodal_inputs: Optional[Dict[str, List[str]]] = None,
        task_messages: Optional[str] = None,
        task_template: Optional[str] = None,
        context_inputs: Optional[Union[str, List[str]]] = None,
        context_cache: Optional[str] = None,
        context_inputs_template: Optional[str] = None,
        system_extra_message: Optional[str] = None,
        include_date: Optional[bool] = False,
        xml_to_dict: Optional[bool] = False,
        model_preference: Optional[str] = None,
        prefilling: Optional[str] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        response_mode: Optional[str] = "plain_response",
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        response_template: Optional[str] = None,
        fixed_messages: Optional[List[Dict[str, Any]]] = None,
        signature: Optional[Union[str, Signature]] = None,
        return_reasoning: Optional[bool] = False,
        temp_team_members: Optional[str] = None,        
        #verbose: Optional[bool] = False,
        description: Optional[str] = None,
        _system_prompt_template: Optional[str] = SYSTEM_PROMPT_TEMPLATE,
        _xml_to_dict_template: Optional[str] = XML_TO_DICT_TEMPLATE,
        _annotations: Optional[Dict[str, type]] = {"message": Union[str, Dict[str, str]], "return": str},
    ):
        super().__init__()

        if stream is True:
            if generation_schema is not None:
                raise ValueError("`generation_schema` is not `stream=True` compatible")

            if output_guardrail is not None:
                raise ValueError("`output_guardrail` is not `stream=True` compatible")

            if response_template is not None:
                raise ValueError("`response_template` is not `stream=True` compatible")

            if xml_to_dict is True:
                raise ValueError("`xml_to_dict=True` is not `stream=True` compatible")

        self._set_xml_to_dict_template(_xml_to_dict_template)

        if signature is not None:
            signature_params = {
                "signature": signature, 
                "instructions": instructions,
                "system_message": system_message,
                "xml_to_dict": xml_to_dict,
            }
            if generation_schema is not None:
                signature_params["generation_schema"] = generation_schema
            self._set_signature(**signature_params)
        else:
            self._set_examples(examples)
            self._set_expected_output(expected_output)        
            self._set_generation_schema(generation_schema)
            self._set_instructions(instructions)
            self._set_system_message(system_message)
            self._set_task_template(task_template)
            self._set_xml_to_dict(xml_to_dict)
            
        self.set_name(name)
        self.set_description(description)
        self._set_annotations(_annotations)
        self._set_context_cache(context_cache)
        self._set_context_inputs(context_inputs)
        self._set_context_inputs_template(context_inputs_template)
        self._set_fixed_messages(fixed_messages)
        self._set_input_guardrail(input_guardrail)
        self._set_output_guardrail(output_guardrail)
        self._set_task_messages(task_messages)
        self._set_model(model)
        self._set_model_preference(model_preference)
        self._set_prefilling(prefilling)
        self._set_system_extra_message(system_extra_message)
        self._set_include_date(include_date)
        self._set_system_prompt_template(_system_prompt_template)
        self._set_response_mode(response_mode)
        self._set_stream(stream)
        self._set_response_template(response_template)
        self._set_return_reasoning(return_reasoning)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_task_inputs(task_inputs)
        self._set_team_members()
        self._set_temp_team_members(temp_team_members)
        self._set_tool_choice(tool_choice)
        self._set_tools(tools)

    def forward(self, message: Union[str, Dict[str, Any], Message], **kwargs):
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(prefilling=self.prefilling, **inputs)
        response = self._process_model_response(message, model_response, **inputs)
        return response

    def _execute_model(
        self, 
        model_state: List[Dict[str, Any]],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None, 
        temp_team_members: Optional[str] = None
    ) -> Union[ModelResponse, ModelStreamResponse]:
        model_execution_params = self._prepare_model_execution(
            model_state, prefilling, model_preference, temp_team_members
        )
        if self.input_guardrail:
            self._execute_input_guardrail(model_execution_params)
        model_response = self.model(**model_execution_params)
        return model_response

    @trace_agent_prepare_model_execution
    def _prepare_model_execution(
        self, 
        model_state: List[Dict[str, Any]],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None, 
        temp_team_members: Optional[str] = None
    ) -> Dict[str, Any]:
        agent_state = []

        if self.fixed_messages:
            agent_state.extend(self.fixed_messages)

        agent_state.extend(model_state)

        system_prompt = self._get_system_prompt(temp_team_members=temp_team_members)

        tool_schemas = self.tool_library.get_tool_json_schemas()
        if not tool_schemas:
            tool_schemas = None

        if is_subclass_of(self.generation_schema, ReAct) and tool_schemas:
            react_tools = get_react_tools_prompt_format(tool_schemas)
            if system_prompt: # TODO: template to react tools
                system_prompt += "\n\n" + react_tools
            else:
                system_prompt = react_tools            
            tool_schemas = None # Disable tool_schemas to react controlflow preference

        model_execution_params = dotdict({
            "messages": agent_state,
            "system_prompt": system_prompt or None,
            "prefilling": prefilling,
            "stream": self.stream,
            "tool_schemas": tool_schemas,
            "tool_choice": self.tool_choice,
            "generation_schema": self.generation_schema,
            "return_reasoning": self.return_reasoning,
            "xml_to_dict": self.xml_to_dict
        })

        if model_preference:
            model_execution_params.model_preference = model_preference

        return model_execution_params

    def _prepare_input_guardrail_execution(
        self, 
        model_execution_params: Dict[str, Any]
    ) -> Dict[str, Any]:
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

    def _process_model_response(
        self, 
        message: Union[str, Dict[str, str], Message],
        model_response: Union[ModelResponse, ModelStreamResponse], 
        model_state: List[Dict[str, Any]],
        model_preference: Optional[str] = None,
        temp_team_members: Optional[str] = None        
    ) -> Union[str, Dict[str, str], Message, ModelStreamResponse]:
        if "tool_call" in model_response.response_type:
            model_response, model_state = self._process_tool_call_response(
                model_response, model_state, model_preference, temp_team_members
            )
        elif is_subclass_of(self.generation_schema, ReAct):
            model_response, model_state = self._process_react_response(
                model_response, model_state, model_preference, temp_team_members
            )
        
        raw_response = self._extract_raw_response(model_response)

        response_type = model_response.response_type

        if response_type in self._supported_outputs:
            response = self._prepare_response(
                raw_response, response_type, model_state, message
            )
            return response
        else:
            raise ValueError(f"Unsupported `response_type={response_type}`")

    def _process_react_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Dict[str, Any],
        model_preference: Optional[str] = None,
        temp_team_members: Optional[str] = None
    ) -> Tuple[Union[str, Dict[str, Any], ModelStreamResponse], Dict[str, Any]]:
        while True:
            raw_response = self._extract_raw_response(model_response)

            if raw_response.current_step:
                actions = raw_response.current_step.actions
                tool_callings = [
                    (act.id, act.name, act.arguments) for act in actions
                ]
                tool_responses = self._process_tool_call(tool_callings)

                for act in actions:
                    act.result = tool_responses[[act.id]]

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

            elif raw_response.final_answer:
                return model_response, model_state

            model_response = self._execute_model(
                model_state=model_state, 
                model_preference=model_preference, 
                temp_team_members=temp_team_members
            )

    def _process_tool_call_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        model_state: Optional[Dict[str, Any]],
        model_preference: Optional[str] = None,
        temp_team_members: Optional[str] = None
    ) -> Tuple[Union[str, Dict[str, Any], ModelStreamResponse], Dict[str, Any]]:
        """
        ToolCall example: [{'role': 'assistant', 'tool_calls': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
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

            model_response = self._execute_model(
                model_state=model_state,
                model_preference=model_preference,
                temp_team_members=temp_team_members
            )

    def _process_tool_call(self, tool_callings: Dict[str, Any]) -> Dict[str, str]: 
        tool_responses = self.tool_library(tool_callings)
        return tool_responses

    def _prepare_response(
        self, 
        raw_response: Union[str, Dict[str, Any], ModelStreamResponse], 
        response_type: str, 
        model_state: Dict[str, Any], 
        message: Union[str, Dict[str, Any], Message]
    ) -> Union[str, Dict[str, Any], ModelStreamResponse]:
        formated_response = None
        if not isinstance(raw_response, ModelStreamResponse):
            if "text_generation" in response_type or "structured" in response_type:
                if self.output_guardrail:
                    self._execute_output_guardrail(raw_response)        
                if self.response_template:
                    formated_response = self._format_response_template(raw_response)
        return self._define_response_mode(formated_response or raw_response, model_state, message)

    def _prepare_output_guardrail_execution(
        self, 
        model_response: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        if isinstance(model_response, str):
            data = model_response
        else:
            data = str(model_response)
        guardrail_params = {"data": data}
        return guardrail_params

    def _define_response_mode(
        self, 
        response: Union[str, Dict[str, Any]], 
        model_state: List[Dict[str, Any]],
        message: Union[str, Dict[str, Any], Message]
    ) -> Union[str, Dict[str, Any], Message, ModelStreamResponse]:
        if self.response_mode == "plain_response":
            return response
        elif self.response_mode == "steps":
            return self._apply_steps_format(model_state, response)
        elif isinstance(message, Message):
            if self.response_mode.startswith(("context", "outputs", "response")):
                message.set(f"{self.response_mode}.{self.name}", response)
            return message
        else:
            raise ValueError(
                "For `response_mode` other than `plain_response` and "
                "`steps` the message object must be of type Message"
            )

    def _apply_steps_format(
        self, 
        model_state: List[Dict[str, Any]],
        response: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        steps_response = chatml_to_steps_format(model_state, response)
        return steps_response

    def _prepare_task(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Prepare model input in ChatML format and execution params."""
        task_messages = None
        runtime_task_messages = kwargs.pop("task_messages", None)        
        
        content = self._process_task_inputs(message, **kwargs)
        
        if isinstance(message, Message):
            task_messages = self._get_task_messages_from_message(message)        
        if runtime_task_messages is not None: # Override with runtime task_messages
            task_messages = runtime_task_messages
        
        if content is None and task_messages is None:
            raise ValueError("No data was detected to make the model input")

        if content is not None:
            chat_content = [{"role": "user", "content": content}]
            if task_messages is None:
                model_state = chat_content
            else:
                task_messages.extend(chat_content)
                model_state = task_messages
        else:
            model_state = task_messages

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        temp_team_members = kwargs.pop("temp_team_members", None)
        if temp_team_members is None and isinstance(message, Message):
            temp_team_members = self._get_temp_team_members_from_message(message)

        return {
            "model_state": model_state,
            "model_preference": model_preference,
            "temp_team_members": temp_team_members
        }

    def _process_task_inputs(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Union[str, Dict[str, Any]]:
        content = ""

        context_content = self._context_manager(message, **kwargs)
        if context_content:
            content += context_content

        if isinstance(message, Message):
            task_inputs = self._extract_message_values(self.task_inputs, message)
        else:
            task_inputs = message

        if task_inputs is None:
            raise AttributeError("When using a `Message` in `nn.Agent` it is necessary to "
                                 "have configured `task_inputs` or `task_template`")

        if self.task_template:
            if task_inputs:
                task_content = self._format_task_template(task_inputs)
            # It's possible to use `task_template` as the default task message
            # if no `task_inputs` is selected. This can be useful for multimodal
            # models that require a text message to be sent along with the data                
            else:                
                task_content = self.task_template
        else:
            task_content = task_inputs

        task_content = apply_xml_tags("task", task_content)
        content += task_content
        content = content.strip() # Remove whitespace
        
        multimodal_content = self._process_task_multimodal_inputs(message, **kwargs)
        if multimodal_content:
            multimodal_content.append({"type": "text", "text": content})
            return multimodal_content
        return content

    def _context_manager(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Optional[str]:
        """Mount context."""
        context_content = ""
        
        if self.context_cache: # Fixed Context Cache
            context_content += self.context_cache        

        context_inputs = None
        runtime_context_inputs = kwargs.pop("context_inputs", None)
        if runtime_context_inputs is not None:
            context_inputs = runtime_context_inputs
        elif isinstance(message, Message):
            context_inputs = self._extract_message_values(self.context_inputs, message)

        if context_inputs is not None:
            if self.context_inputs_template:
                msg_context = self._format_template(context_inputs, self.context_inputs_template)
            else:
                if isinstance(context_inputs, str):
                    msg_context = context_inputs
                elif isinstance(context_inputs, list):
                    msg_context = " ".join(str(v) for v in context_inputs if v is not None)
                elif isinstance(context_inputs, dict):
                    msg_context = "\n\n".join(str(v) for v in context_inputs.values())                
            context_content += "\n\n" + msg_context
            
        if context_content:
            return apply_xml_tags("context", context_content)
        return None

    def _process_task_multimodal_inputs(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Processes multimodal inputs (image, audio, file) via kwargs or message.
        Returns a list of multimodal content in ChatML format.
        """
        multimodal_paths = None
        task_multimodal_inputs = kwargs.get("task_multimodal_inputs", None)
        if task_multimodal_inputs is not None:
            multimodal_paths = task_multimodal_inputs
        elif isinstance(message, Message) and self.task_multimodal_inputs is not None:
            multimodal_paths = self._extract_message_values(self.task_multimodal_inputs, message)

        if multimodal_paths is None:
            return None

        content = []
        
        formatters = {
            "image": self._format_image_input,
            "audio": self._format_audio_input,
            "file": self._format_file_input,
        }

        for media_type, formatter in formatters.items():
            media_sources = multimodal_paths.get(media_type, [])
            if not isinstance(media_sources, list):
                logger.warning(f"Expected list for multimodal config key `{media_type}`, "
                               f"got `{type(media_sources)}`")
                continue

            for media_source in media_sources:
                if media_source:
                    formatted_input = formatter(media_source)
                    if formatted_input:
                        content.append(formatted_input)

        return content

    def _format_image_input(self, image_source: str) -> Optional[Dict[str, Any]]:
        """Formats the image input for the model"""
        base64_image = self._prepare_data_uri(image_source, force_encode=True)

        if not base64_image:
            return None

        mime_type = get_mime_type(image_source) # Try to guess from the original source
        if not mime_type.startswith("image/"): mime_type = "image/jpeg" # Fallback        
        image_data_url = f"data:{mime_type};base64,{base64_image}"
        
        return {"type": "image_url", "image_url": {"url": image_data_url}}

    def _format_audio_input(self, audio_source: str) -> Optional[Dict[str, Any]]:
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

    def _format_file_input(self, file_source: str) -> Optional[Dict[str, Any]]:
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

    def _get_temp_team_members_from_message(self, message: Message) -> Optional[List[str]]:
        """Extract temp_team_members from Message if configured."""
        return self._get_content_from_message(self.temp_team_members, message)

    def _get_task_messages_from_message(self, message: Message) -> Optional[List[Dict[str, Any]]]:
        """Returns a message history (ChatML format) from message"""        
        return self._get_content_from_message(self.task_messages, message)

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

    def _set_context_inputs_template(self, context_inputs_template: Optional[str] = None):
        if isinstance(context_inputs_template, str) or context_inputs_template is None:
            self.register_buffer("context_inputs_template", context_inputs_template)
        else:
            raise TypeError("`context_inputs_template` requires a string or None"
                            f"given `{type(context_inputs_template)}`")

    def _set_prefilling(self, prefilling: Optional[str] = None):
        if isinstance(prefilling, str) or prefilling is None:
            self.register_buffer("prefilling", prefilling)
        else:
            raise TypeError("`prefilling` requires a string or None"
                            f"given `{type(prefilling)}`")        

    def _set_response_mode(self, response_mode: str):
        if isinstance(response_mode, str):
            if (
                response_mode in ["plain_response", "steps", "response"] # deprecated
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

    def _set_return_reasoning(self, return_reasoning: bool):
        if isinstance(return_reasoning, bool):
            self.register_buffer("return_reasoning", return_reasoning)
        else:
            raise TypeError("`return_reasoning` requires a bool "
                            f"given `{type(return_reasoning)}`")

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
            raise TypeError("`generation_schema` need be a `msgspec.Struct` or None "
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

    def _set_include_date(self, include_date: Optional[bool] = False):
        if isinstance(include_date, bool):
            self.register_buffer("include_date", include_date)
        else:
            raise TypeError("`include_date` requires a bool "
                            f"given `{type(include_date)}`")

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

    def _set_team_members(self, team_members: Optional[str] = None):
        if isinstance(team_members, str) or team_members is None:
            self.register_buffer("team_members", team_members)
        else:
            raise TypeError("`team_members` requires a string or None "
                            f"given `{type(team_members)}`")

    def _set_temp_team_members(self, temp_team_members: Optional[str] = None):
        if isinstance(temp_team_members, str) or temp_team_members is None:
            self.register_buffer("temp_team_members", temp_team_members)
        else:
            raise TypeError("`temp_team_members` requires a string or None "
                            f"given `{type(temp_team_members)}`")

    def _set_system_prompt_template(self, system_prompt_template: Optional[str] = None):
        if isinstance(system_prompt_template, str) or system_prompt_template is None:
            self.register_buffer("system_prompt_template", system_prompt_template)
        else:
            raise TypeError("`system_prompt_template` requires a string given "
                            f"`{type(system_prompt_template)}`")

    def _set_system_extra_message(self, system_extra_message: Optional[str] = None):
        if isinstance(system_extra_message, str) or system_extra_message is None:
            self.register_buffer("system_extra_message", system_extra_message)
        else:
            raise TypeError("`system_extra_message` requires a string or None "
                            f"given `{type(system_extra_message)}`")

    def _set_xml_to_dict_template(self, xml_to_dict_template: str):
        if isinstance(xml_to_dict_template, str):
            self.register_buffer("xml_to_dict_template", xml_to_dict_template)
        else:
            raise TypeError("`xml_to_dict_template` requires a string "
                            f"given `{type(xml_to_dict_template)}`")        

    def _set_xml_to_dict(self, xml_to_dict: Optional[bool] = False):
        if isinstance(xml_to_dict, bool):
            if xml_to_dict:
                json_schema = None
                if self.generation_schema:
                    schema = msgspec.json.schema(self.generation_schema)                  
                    json_schema = adapt_struct_schema_to_json_schema(schema)                    
                template_inputs = {
                    "instructions": self.instructions.data,
                    "json_schema": json_schema
                }
                xml_instructions = self._format_template(
                    template_inputs, self.xml_to_dict_template
                )
                self._set_instructions(xml_instructions)
            self.register_buffer("xml_to_dict", xml_to_dict)
        else:
            raise TypeError(f"`xml_to_dict` requires a bool given `{type(xml_to_dict)}`")

    def _set_signature(
        self, 
        signature: Optional[Union[str, Signature]] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        instructions: Optional[str] = None,
        system_message: Optional[str] = None,
        xml_to_dict: Optional[bool] = False
    ):
        if signature is not None:

            examples = None

            # Get system message
            schema_system_message = SIGNATURE_SYSTEM_MESSAGES.get(generation_schema, SIGNATURE_DEFAULT_SYSTEM_MESSAGE)
            self._set_system_message(system_message or schema_system_message)

            if isinstance(signature, str):
                input_str_signature, output_str_signature = signature.split("->")  
                inputs_desc = StructFactory._parse_annotations(input_str_signature)
                outputs_desc = StructFactory._parse_annotations(output_str_signature)

            elif issubclass(signature, Signature):
                # Get instructions
                instructions = signature.get_instructions()

                # Get examples from signature
                examples = get_examples_from_signature(signature)

                # Descriptions
                inputs_desc = signature.get_input_descriptions()
                outputs_desc = signature.get_output_descriptions()
                output_str_signature = signature.get_str_signature().split("->")[-1]

            else:
                raise TypeError("`signature` requires a string, `Signature` or None "
                                f"given `{type(signature)}`")
            
            # Create task template
            task_template = get_task_template_from_signature(inputs_desc)
            self._set_task_template(task_template)

            # Set instructions
            self._set_instructions(instructions)

            # Create generation schema
            output_struct = StructFactory.from_signature(output_str_signature, "Outputs")
            if generation_schema is not None:            
                output_struct = generation_schema[output_struct] # Insert as an TypeVar
                class Output(output_struct, msgspec.Struct): # Convert typing._GenericAlias to Struct
                    pass
                output_struct = Output
            self._set_generation_schema(output_struct)

            # Create expected outputs
            expected_output = get_expected_output_from_signature(inputs_desc, outputs_desc)
            if xml_to_dict is False:
                expected_output += "\nWrite an encoded JSON."
            self._set_expected_output(expected_output)

            # Create examples
            if examples is not None:
                input_examples_dict, output_json_string = examples
                input_examples_string = self._format_task_template(input_examples_dict, xml_to_dict)
                examples = format_examples([(input_examples_string, output_json_string)])
            self._set_examples(examples)

            # Set xml output
            self._set_xml_to_dict(xml_to_dict)

    def _get_system_prompt(self, temp_team_members=None) -> str:
        """
        Render the system prompt using the Jinja template.
        Returns an empty string if no segments are provided.
        """
        template_inputs = {
            "system_message": self.system_message.data,
            "instructions": self.instructions.data,
            "expected_output": self.expected_output.data,
            "examples": self.examples.data,
            "system_extra_message": self.system_extra_message,
        }

        if self.include_date:
            template_inputs["current_date"] = datetime.now().strftime("%m/%d/%Y")
                
        combined_team_members = [] # Combine team_members with temp_team_members
        if self.team_members:
            combined_team_members.extend(self.team_members)
        if temp_team_members:
            combined_team_members.extend(temp_team_members)
        
        if combined_team_members:
            template_inputs["team_members"] = combined_team_members
            
        system_prompt = self._format_template(
            template_inputs, self.system_prompt_template
        )
        return system_prompt
