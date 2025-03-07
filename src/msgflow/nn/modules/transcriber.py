from typing import Dict, Optional, Union
from msgflow.message import Message
from msgflow.models.router import ModelRouter
from msgflow.models.types import ASRModel
from msgflow.nn.modules.module import Module
from msgflow.utils.encode import to_bytes


class Transcriber(Module):
    """Transcriber is a Module type that uses language models to transcribe audios.

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
        language: Spoken language acronym.
        response_format: How the model should format the output. Options:
            * json
            * text (default)
            * srt
            * verbose_json
            * vtt
        timestamp_granularities: Enable timestamp granularities.
            Requires `response_format=verbose_json`. Options:
            * word
            * segment
            * None (default)
        prompt: Useful for instructing the model to follow some transcript generation pattern.
    """

    def __init__(
        self,
        name: str,
        model: Union[ASRModel, ModelRouter],
        *,        
        task_inputs: Union[str, Dict[str, str]] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = "text",
        timestamp_granularities: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__()
        self.set_name(name)
        self._set_model(model)
        self._set_task_inputs(task_inputs)
        self._set_response_mode(response_mode)
        self._set_language(language)
        self._set_response_format(response_format)
        self._set_timestamp_granularities(timestamp_granularities)
        self._set_prompt(prompt)
        self._set_response_template(response_template)

    def forward(self, message: Union[str, Message]):
        audio = self._prepare_task(message)
        model_response = self._execute_model(audio)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(self, audio):
        model_response = self.model(
            audio=audio,
            language=self.language,
            response_format=self.response_format,
            timestamp_granularities=self.timestamp_granularities,
            prompt=self.prompt,
        )
        return model_response

    def _process_model_response(self, model_response, message):
        if model_response.response_type == "transcript":
            raw_response = model_response.consume()
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    def _prepare_task(self, message):
        if isinstance(message, str):
            audio_data = message
        elif isinstance(message, Message):
            audio_data = self._process_message_task(message)
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        
        audio = to_bytes(audio_data)
        
        return audio

    def _process_message_task(self, message: Message):
        if isinstance(self.task_inputs, tuple):
            content = self._get_content_from_or_input(self.task_inputs, message)
        else:
            content = message.get(self.task_inputs)

        if content is None:
            raise ValueError(f"No audio found in paths: {self.task_inputs}")

        return content

    def _set_model(self, model: Union[ASRModel, ModelRouter]):
        if isinstance(model, ASRModel) or isinstance(model, ModelRouter):
            self.register_buffer("model", model)
        else:
            raise TypeError("`model` need be a `ASRModel` or `ModelRouter` "
                             f"given `{type(model)}")

    def _set_language(self, language: Optional[str] = None):
        if isinstance(language, str) or language is None:
            self.register_buffer("language", language)
        else:
            raise TypeError(f"`language` need be a `str` or `None` given `{type(language)}")

    def _set_timestamp_granularities(self, timestamp_granularities: str):
        if isinstance(timestamp_granularities, str):
            supported_granularities = ["word", "segment"]
            if timestamp_granularities in supported_granularities:
                timestamp_granularities = [timestamp_granularities]
            else:
                raise ValueError(f"`timestamp_granularities` can be {supported_granularities} "
                                 f"given {timestamp_granularities}")
        elif timestamp_granularities is not None:
            raise TypeError("`timestamp_granularities` need be a `str` or `None` "
                            f"given `{type(timestamp_granularities)}")    
        self.register_buffer("timestamp_granularities", timestamp_granularities)        

    def _set_response_format(self, response_format: str):
        supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
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

    def _set_prompt(self, prompt: Optional[str] = None):
        if isinstance(prompt, str) or prompt is None:
            self.register_buffer("prompt", prompt)
        else:
            raise TypeError(f"`prompt` need be a str or None given `{type(prompt)}")            
