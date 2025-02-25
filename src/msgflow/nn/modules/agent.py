from enum import Enum
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
import gevent
import msgspec

from msgflow.models.router import ModelRouter
from msgflow.models.response import Response, StreamResponse
from msgflow.message import Message
from msgflow.models.types import ChatCompletionModel
from msgflow.utils import (
    chatml_to_steps_format,
    encode_local_file_in_base64,
    encode_base64_from_url,
    is_base64,
)
from msgflow.utils.validation import is_subclass_of
from msgflow.generation.reasoning.react import ReAct
from msgflow.nn.modules.module import Module
from msgflow.nn.modules.tool import ToolLibrary
from msgflow.nn.parameter import Parameter

# Context Manager function that will manage the processing of assembling the context
# new features: context_cache fixed message in the model, in the retrieval
# initial_assist_msg or prefix_agent_msg this will be a param (prefilling)
# it is possible to continue generating a model. Just resend to it what it
# wrote and then it will continue from there

# from schema to scheme

# if not all(isinstance(module, Module) for module in modules_to_send):

# the system can change the response to the stream if x condition is met. nein

# add time/date to the system prompt (this can be bad if you use prompt cache)

class PromptSpec(Enum):
    SYSTEM_PROMPT = "Who are you"
    INSTRUCTIONS = "How you should do"
    # FEW_SHOT = 'Samples of what to do'
    EXPECTED_OUTPUT = "Describes what the response should be like"


class Agent(Module):
    r"""Agent is a Module type that uses language models to solve tasks.

    An Agent can perform actions in an environment using function calls.
    For an Agent, a function is any callable object.

    An Agent can handle multimodal inputs and outputs.

    Args:
        name: Agent name in snake case format.
        model: ChatCompletation Model client.
        system_prompt: The Agent behaviour.
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
        tool_library:
            ...
        response_template:
            ...
        prefilling:
            ...
        context_cache:
            ...
        chat_history:
            ...
        chat_history_mode:
            ...
        chat_few_shot
            ...
        audio_input_format
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
        "b64_audio_generation",
        "b64_audio_text_generation",
    ]

    def __init__(
        self,
        name: str,
        model: Union[ChatCompletionModel, ModelRouter],
        *,
        system_prompt: Optional[str] = "",
        instructions: Optional[str] = "",
        expected_output: Optional[str] = "",
        stream: Optional[bool] = False,
        task_inputs: Optional[Union[str, Dict[str, str]]] = None,
        task_template: Optional[str] = None,
        task_multimodal_inputs: Optional[Dict[str, List[str]]] = None,
        context_inputs: Optional[Union[str, List[str]]] = None,
        prefilling: Optional[str] = None, # NEW
        context_cache: Optional[str] = None, # NEW
        generation_schema: Optional[msgspec.Struct] = None,
        response_mode: Optional[str] = "plain_response",
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        response_template: Optional[str] = "",
        # chat_history: Optional[Union[ChatHistory, MultiChatHistory]] = None, #TODO: requires reason here
        # chat_history_mode: Literal["relevant", "recent", "full"] = "relevant",
        chat_few_shot: Optional[List[Dict[str, Any]]] = None,
        predicted_outputs: Optional[bool] = False,
        signature: Optional[str] = None,
        audio_input_format: Optional[Literal["standard", "generation"]] = "generation",
        verbose: Optional[bool] = False,
        description: Optional[str] = "",
        _annotations: Optional[Dict[str, type]] = None,
    ):
        super().__init__()

        if stream and response_template:
            raise ValueError("`response_template` is not `stream=True` compatible")
        if stream and issubclass(generation_schema, ReAct):
            raise ValueError(
                "`generation_schema=ReAct` is not `stream=True` compatible"
            )
        if tools and predicted_outputs:
            raise ValueError("`tools` is not `predicted_outputs=True` compatible")
        if task_template is None and predicted_outputs:
            raise ValueError("`predicted_outputs=True` requires a `task_template`")

        self.set_name(name)
        self.set_description(description)
        self._set_model(model)
        self._set_expected_output(expected_output)
        self._set_instructions(instructions)
        self._set_system_prompt(system_prompt)
        self._set_generation_schema(generation_schema)
        self._set_audio_input_format(audio_input_format)
        self._set_chat_few_shot(chat_few_shot)
        # self._set_chat_history(chat_history)
        self._set_context_inputs(context_inputs)
        self._set_context_cache(context_cache)
        self._set_prefilling(prefilling)
        self._set_tools(tools)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_task_inputs(task_inputs)
        self._set_task_template(task_template)
        self._set_stream(stream)
        self._set_tool_choice(tool_choice)
        self._set_predicted_outputs(predicted_outputs)
        self._set_annotations(_annotations or {"message": str, "return": str})

    def forward(self, message: Union[str, Dict[str, Any], Message]):
        model_state = self._prepare_task(message)    
        model_response = self._execute_model(model_state, self.prefilling)
        response = self._process_model_response(model_response, model_state, message)
        return response

    def _execute_model(self, model_state, prefilling=None):
        agent_state = []

        agent_system_prompt = self._get_agent_system_prompt()

        if agent_system_prompt:
            agent_state.extend(agent_system_prompt)

        if self.tool_library:
            tool_schemas = self.tool_library.get_functions_json_schema()
        else:
            tool_schemas = None

        if issubclass(self.generation_schema, ReAct) and tool_schemas:
            react_tools = f"## Available tools: \n{tool_schemas}"
            if agent_system_prompt:
                agent_system_prompt += f"\n\n {react_tools}"
            else:
                agent_system_prompt = react_tools
            # Disable tool_schemas to react controlflow preference
            tool_schemas = None
        
        if self.chat_few_shot:
            agent_state.extend(self.chat_few_shot)

        agent_state.extend(model_state)

        if self.verbose:
            print(agent_state)

        model_response = self.model(
            messages=agent_state,
            system_prompt=agent_system_prompt,
            prefilling=prefilling,
            stream=self.stream,
            tool_schemas=tool_schemas,
            tool_choice=self.tool_choice,
            generation_schema=self.generation_schema,
        )        

        return model_response

    def _process_model_response(self, model_response, model_state, message):
        if model_response.response_type == "tool_call":
            raw_response, model_response, model_state = (
                self._process_tool_call_response(model_response, model_state)
            )
        elif issubclass(self.generation_schema, ReAct):
            raw_response, model_response, model_state = self._process_react_response(
                model_response, model_state
            )
        elif isinstance(model_response, Response):
            raw_response = model_response.consume()
        elif isinstance(model_response, StreamResponse):
            raw_response = model_response
        else:
            raise ValueError(f"Unsupported `model_response={type(model_response)}`")

        if model_response.response_type in self._supported_outputs:
            response = self._prepare_response(
                # TODO here instead of passing just the response type
                # I can pass the entire model_response and then
                # capture the metadata using trace
                raw_response, model_response.response_type, message, model_state
            )
            return response
        else:
            raise ValueError(
                f"Unsupported `response_type={model_response.response_type}`"
            )

    def _process_react_response(self, model_response, model_state):
        while True:
            raw_response = model_response.consume()

            if raw_response.get("current_step"):
                actions = raw_response["current_step"]["actions"]
                tool_callings = [
                    (act["id"], act["name"], act["arguments"]) for act in actions
                ]
                tool_results = self._process_tool_call(tool_callings)

                for act in actions:
                    act["result"] = tool_results[[act["id"]]]

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
                return raw_response, model_response, model_state

            model_response = self._execute_model(model_state)

    def _process_tool_call_response(self, model_response, model_state):
        """
        Mensagens: [{'role': 'assistant', 'tool_calls': [{'id': 'call_1YLHAVwHwDPjEBuMpWQfSktO',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_1YLHAVwHwDPjEBuMpWQfSktO',
        'content': '2024-10-15'}]
        """
        while True:
            if model_response.response_type == "tool_call":
                response = model_response.consume()
                tool_callings = response.get_calls()
                tool_results = self._process_tool_call(tool_callings)
                response.insert_results(tool_results)
                tool_results_message = response.get_messages()
                model_state.extend(tool_results_message)
            else:
                raw_response = model_response.consume()
                return raw_response, model_response, model_state

            model_response = self._execute_model(model_state)

    def _process_tool_call(self, tool_callings):
        greenlet = gevent.spawn(self.tool_library, tool_callings)
        tool_results = greenlet.get()
        return tool_results

    def _prepare_response(self, raw_response, response_type, message, model_state):
        if (
            response_type == "structured"
            and self.response_mode != "steps"
            and raw_response.get("final_answer")
        ):
            raw_response = raw_response["final_answer"]

        if self.response_template and response_type in [
            "text_generation",
            "structured",
        ]:
            # TODO: support to b64_audio_text_generation, add template in transcript
            response = self._format_response_template(raw_response)
        else:
            response = raw_response

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

    def _apply_steps_format(self, model_state, response):
        steps_response = chatml_to_steps_format(model_state, response)
        return steps_response

    def _prepare_task(
        self, message: Union[str, Dict[str, Any], Message]
    ) -> List[Dict[str, Any]]:
        if isinstance(message, (str, dict)):
            content = self._process_str_dict_task(message)
        elif isinstance(message, Message):
            content = self._process_message_task(message)
        else:
            raise ValueError("Unsupported message type")
        return [{"role": "user", "content": content}]

    def _process_str_dict_task(self, message: str) -> List[Dict[str, Any]]:
        content = self._format_task_template(message)
        return content

    def _process_message_task(self, message: Message) -> List[Dict[str, Any]]:
        content = ""

        # Process context
        context_content = self._process_context(message)

        if context_content:
            content += f"# Context:\n{context_content}\n\n"

        # Process text content
        if self.task_inputs:
            text_content = self._process_text_inputs(message)

            if self.task_template:
                text_content = self._format_task_template(text_content)
            content += f"# Task:\n{text_content}\n\n"

        # It's possible to use `task_template` as the default task message
        # if no `task_inputs` is selected. This can be useful for multimodal
        # models that require a text message to be sent along with the data
        elif self.task_template:
            content += f"# Task:\n{self.task_template}\n\n"

        # Remove whitespace
        content = content.strip()

        # Process multimodal content
        if self.task_multimodal_inputs:
            multimodal_content = []
            multimodal_content.append({"type": "text", "text": content})
            multimodal_content.extend(self._process_multimodal_inputs(message))
            return multimodal_content
        else:
            return content

    def _process_text_inputs(self, message: Message) -> Union[str, Dict[str, Any]]:
        # TODO allow other fields besides outputs?
        if self.task_inputs == "outputs":  # Consume all values in outputs
            return "\n\n".join(str(v) for v in message.get("outputs").values())
        elif isinstance(self.task_inputs, str):
            return message.get(self.task_inputs)
        elif isinstance(self.task_inputs, dict):
            text_inputs = {}
            for k, v in self.task_inputs.items():                
                if isinstance(v, tuple): # OR inputs
                    text_inputs[k] = self._get_content_from_or_input(v, message)
                else:
                    text_inputs[k] = message.get(v)
            return text_inputs
        else:
            return None

    def _process_context(self, message: Message): # TODO support to list, dict, str
        if not self.context_inputs:
            return None

        if isinstance(self.context_inputs, str) and self.context_inputs == "context":
            return "\n\n".join(str(v) for v in message.get("context").values())
        elif isinstance(self.context_inputs, list):  # ["context.1", "context.2"]
            context_values = []
            for path in self.context_inputs:
                # OR inputs
                if isinstance(path, tuple):
                    context_value = self._get_content_from_or_input(path, message)
                    if context_value is not None:
                        context_values.append(str(context_value))
                else:
                    context_values.append(str(message.get(path)))
            return " ".join(context_values)
        else:
            raise ValueError("Invalid context_inputs format")

    def _process_multimodal_inputs(self, message: Message) -> List[Dict[str, Any]]:
        # TODO: suporte para consumir todas as entradas de images or outro
        content = []
        for image_path in self.task_multimodal_inputs.get("images", []):
            if isinstance(image_path, tuple):
                image_data = self._get_content_from_or_input(image_path, message)
            else:
                image_data = message.get(image_path)
            if image_data:
                if not image_data.startswith("http") and not is_base64(image_data):
                    base64_image = encode_local_file_in_base64(image_data)
                    image_data = f"data:image/jpeg;base64,{base64_image}"
                content.append({"type": "image_url", "image_url": {"url": image_data}})

        for audio_path in self.task_multimodal_inputs.get("audios", []):
            if isinstance(audio_path, tuple):
                image_data = self._get_content_from_or_input(audio_path, message)
            else:
                audio_data = message.get(audio_path)
            if audio_data:
                audio_format = Path(audio_data).suffix
                if not audio_data.startswith("http") and not is_base64(audio_data):
                    base64_audio = encode_local_file_in_base64(audio_data)
                if self.audio_input_format == "standard":  # vLLM style
                    if not is_base64(audio_data):
                        audio_data = f"data:audio/{audio_format};base64,{base64_audio}"
                    content.append(
                        {"type": "audio_url", "audio_url": {"url": audio_data}}
                    )
                elif self.audio_input_format == "generation":  # OpenAI style
                    # OpenAI requires a base64 file as input
                    if audio_data.startswith("http"):
                        audio_data = encode_base64_from_url(audio_data)
                    content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_data, "format": audio_format},
                        }
                    )
        return content

    # TODO move to module
    def _set_task_template(self, task_template: Optional[str] = None):        
        if isinstance(task_template, str) or task_template is None:
            if isinstance(task_template, str) and task_template == "":
                raise ValueError("`task_template` requires a string not empty "
                                 f"given {task_template}")
            self.register_buffer("task_template", task_template)
        else:
            raise TypeError("`task_template` requires a string or None "
                            f"given `{type(template)}`")

    def _set_task_inputs(self, task_inputs: Optional[Union[str, Dict[str, str]]] = None):
        # TODO: suporte para lista de inputs ["outputs.text1", "outputs.text2"]
        if isinstance(task_inputs, (str, dict)) or task_inputs is None:
            if isinstance(task_inputs, str) and task_inputs == "":
                raise ValueError("`task_inputs` requires a string not empty " 
                                 f"given `{task_inputs}`")
            if isinstance(task_inputs, dict) and not task_inputs:
                raise ValueError("`task_inputs` requires a dict not empty "
                                 f"given `{task_inputs}`")                                 
            self.register_buffer("task_inputs", task_inputs)
        else:
            raise TypeError("`task_inputs` requires a string, dict or None, "
                            f"given `{type(task_inputs)}`")

    def _set_task_multimodal_inputs(
        self, 
        task_multimodal_inputs: Optional[Dict[str, List[str]]] = None
    ):        
        # TODO permitir passar em vez de uma lista passar so um valor se for unico
        if isinstance(task_multimodal_inputs, dict):
            if not task_multimodal_inputs:
                raise ValueError("`task_multimodal_inputs` requires a dict not empty"
                                 f"given `{task_multimodal_inputs}`")
            self.register_buffer("task_multimodal_inputs", task_multimodal_inputs)
        else:
            raise TypeError("`task_multimodal_inputs` requires a dict "
                            f"given `{type(task_multimodal_inputs)}`")

    def _set_audio_input_format(self, audio_input_format: str):
        if isinstance(audio_input_format, str):
            if audio_input_format not in ["standard", "generation"]:
                raise ValueError(
                    "`audio_input_format` must be either `standard` or `generation`"
                    f"given `{audio_input_format}`"
                )
            self.register_buffer("audio_input_format", audio_input_format)
        else:
            raise TypeError("`audio_input_format` requires a string"
                            f"given `{type(audio_input_format)}`")

    def _set_context_inputs(self, context_inputs: Optional[Union[str, List[str]]] = None):
        if isinstance(context_inputs, (str, list)) or context_inputs is None:
            if isinstance(context_inputs, str) and context_inputs == "":
                raise ValueError("`context_inputs` requires a string not empty"
                                 f"given `{context_inputs}"`)
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
            if response_mode in [
                "plain_response",
                "steps",
                "response",
            ] or response_mode.startswith(("context", "outputs")):
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
        if (isinstance(tools, list) and all(callable(obj) for obj in tools)) or None:                
            tool_library = ToolLibrary(tools)
            self.tool_library = tool_library
        else:
            raise TypeError("`tool_library` need be a list of callables or None"
                            f"given `{type(tool_library)}`")

    def _set_stream(self, stream: bool):
        if isinstance(stream, bool):
             self.register_buffer("stream", stream)
        else:
            raise TypeError(f"`stream` need be a bool given `{type(stream)}`")            

    # TODO: chat_few_shot eh um nome meio ruim
    def _set_chat_few_shot(self, chat_few_shot: Optional[List[Dict[str, Any]]] = None):
        if (isinstance(chat_few_shot, list) and all(dict(obj) for obj in chat_few_shot)) or None: 
            self.register_buffer("chat_few_shot", chat_few_shot)
        else:
            raise TypeError("`chat_few_shot` need be a list of dict or None"
                            f"given `{type(chat_few_shot)}`")

    def _set_generation_schema(self, generation_schema: Optional[msgspec.Struct] = None):
        if is_subclass_of(generation_schema, msgspec.Struct) or generation_schema is None:
            self.register_buffer("generation_schema", generation_schema)
        else:
            raise TypeError("`generation_schema` need be a `msgspec.Struct` or None"
                            f"given `{type(generation_schema)}`")

    def _set_model(self, model: Union[ChatCompletionModel, ModelRouter]):
        if isinstance(model, (ChatCompletionModel, ModelRouter)):
            self.register_buffer("model", model)
        else:
            raise TypeError("`model` need be a `ChatCompletionModel` "
                            f"or `ModelRouter` given `{type(model)}`")

    # TODO 
    #def _set_predicted_outputs(self, predicted_outputs: bool):
    #    if isinstance(predicted_outputs, bool):
    #        self.register_buffer("predicted_outputs", predicted_outputs)
    #    else:
    #        raise TypeError("`predicted_outputs` requires bool "
    #                        f"given `{type(predicted_outputs)}`")

    def _set_tool_choice(self, tool_choice: Optional[str] = None):
        if isinstance(tool_choice, str) or tool_choice is None:
            if isinstance(tool_choice, str):
                if tool_choice not in ["auto", "required"]:
                    tool_choice = {"type": "function", "function": {"name": tool_choice}}
            self.register_buffer("tool_choice", tool_choice)
        else:
            raise TypeError("`tool_choice` need be a str or None "
                            f"given `{type(tool_choice)}`")            

    # TODO move to module
    def _set_response_template(self, response_template: Optional[str] = None):
        if isinstance(response_template, str) or response_template is None:
            if isinstance(response_template, str) and response_template == "":
                raise ValueError("`response_template` cannot be an empty str")
            self.register_buffer("response_template", response_template)
        else:
            raise TypeError("`response_template` requires a string or None "
                            f"given `{type(response_template)}`")

    def _set_system_prompt(self, system_prompt: Optional[str] = None):
        if isinstance(system_prompt, str) or system_prompt is None:
            self.system_prompt = Parameter(system_prompt, PromptSpec.SYSTEM_PROMPT)
        else:
            raise TypeError("`system_prompt` requires a string or None "
                            f"given `{type(system_prompt)}`")

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

    def _get_agent_system_prompt(self):    
        agent_system_prompt = ""

        if self.system_prompt.data:
            agent_system_prompt += f"{self.system_prompt.data}\n\n"

        if self.instructions.data:
            agent_system_prompt += f"# Instructions:\n{self.instructions.data}\n\n"

        if self.expected_output.data:
            agent_system_prompt += (
                f"# Expected Output:\n{self.expected_output.data}\n\n"
            )

        return agent_system_prompt