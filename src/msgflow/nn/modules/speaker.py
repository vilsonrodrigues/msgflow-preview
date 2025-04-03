from typing import Dict, Literal, Optional, Union
from msgflow.message import Message
from msgflow.models.gateway import ModelGateway
from msgflow.models.types import TTSModel
from msgflow.nn.modules.module import Module


class Speaker(Module):
    """Speaker is a Module type that uses language models to transform text in speak.

    Args:
        name: Transcriber name in snake case format.
        model: Transcriber Model client.
        task_inputs: Fields of the Message object that will be the input to the task.
        response_mode: What the response should be. Has five options:
            * `plain_response` (default): Returns the transcriber response.
            * `response`: Write on `response` field in Message object.
            * `context`: Write on `context` field in Message object.
               It`s insert how `context.transcriber_name`.
            * `outputs`: Write on `outputs` field in Message object.
               It`s insert how `outputs.transcriber_name`.

        prompt: Useful for instructing the model to follow some speak generation pattern.
        response_format: ["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "opus"
        prompt: ...
    """

    def __init__(
        self,
        name: str,
        model: Union[TTSModel, ModelGateway],
        *,      
        stream: Optional[bool] = False,
        task_inputs: Optional[str] = None,
        response_mode: Optional[str] = "plain_response",
        response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "opus",
        prompt: Optional[str] = None,
    ):
        super().__init__()
        self.set_name(name)    
        self._set_model(model)
        self._set_prompt(prompt)
        self._set_response_format(response_format)
        self._set_response_mode(response_mode)
        self._set_stream(stream)
        self._set_task_inputs(task_inputs)

    def forward(self, message: Union[str, Message]):
        text = self._prepare_task(message)
        model_response = self._execute_model(text)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(self, text):
        model_response = self.model(
            text=text,
            response_format=self.response_format.data,
            prompt=self.prompt.data,
            stream=self.stream.data            
        )
        return model_response

    def _process_model_response(self, model_response, message):
        if model_response.response_type == "audio_generation":
            raw_response = self._extract_raw_response(model_response)
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    def _prepare_response(self, raw_response, message):
        return self._define_response_mode(raw_response, message)

    def _prepare_task(self, message):
        if isinstance(message, str):
            text = message
        elif isinstance(message, Message):
            text = self._process_message_task(message)
        else:
            raise ValueError(f"Unsupported message type: `{type(message)}`")
                
        return text

    def _process_message_task(self, message: Message):         
        content = self._process_text_inputs(message)
        return content

    def _process_text_inputs(self, message: Message):
        if isinstance(self.task_inputs.data, str):
            content = message.get(self.task_inputs.data)
        elif isinstance(self.task_inputs.data, tuple): # OR inputs
            content = self._get_content_from_or_input(self.task_inputs.data, message)

        if content is None:
            raise ValueError(f"No text found in paths: `{self.task_inputs.data}`")
        return content

    def _set_model(self, model: Union[TTSModel, ModelGateway]):
        if (
            isinstance(model, TTSModel) 
            or 
            (isinstance(model, ModelGateway) and model.model_types == "tts")
        ):
            self.register_buffer("model", model)
        else:
            raise TypeError("`model` need be a `TTSModel` or `ModelGateway` "
                             f"given `{type(model)}")

    def _set_response_format(self, response_format: str):
        supported_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if isinstance(response_format, str):
            if response_format in supported_formats:
                self.register_buffer("response_format", response_format)
            else:
                raise ValueError(
                    f"`response_format` can be `{supported_formats}` "
                    f"given `{response_format}"
                )    
        else:
            raise TypeError(f"`response_format` need be a str or given `{type(response_format)}")               
