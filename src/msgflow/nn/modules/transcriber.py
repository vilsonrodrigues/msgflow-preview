from typing import Dict, Optional, Union
from msgflow.message import Message
from msgflow.models.gateway import ModelGateway
from msgflow.models.types import ASRModel
from msgflow.nn.modules.module import Module
from msgflow.utils.encode import encode_data_to_bytes


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
        model: Union[ASRModel, ModelGateway],
        *,      
        stream: Optional[bool] = False,
        task_multimodal_inputs: Optional[Dict[str, str]] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = "text",
        timestamp_granularities: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__()
        self.set_name(name)
        self._set_language(language)        
        self._set_model(model)
        self._set_prompt(prompt)
        self._set_response_format(response_format)        
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)     
        self._set_stream(stream)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_timestamp_granularities(timestamp_granularities)

    def forward(self, message: Union[str, Message]):
        model_preference = self.get_model_preference(message)
        data = self._prepare_task(message)
        model_response = self._execute_model(data, model_preference)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(self, data, model_preference=None):
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = self.model.data(**model_execution_params)
        return model_response

    def _prepare_model_execution(self, data, model_preference=None):
        model_execution_params = {
            "data": data,
            "language": self.language.data,
            "response_format": self.response_format.data,
            "timestamp_granularities": self.timestamp_granularities.data,
            "prompt": self.prompt.data,
            "stream": self.stream.data,
        }
        if model_preference:
            model_execution_params["model_preference"] = model_preference   
        return model_execution_params

    def _process_model_response(self, model_response, message):
        if model_response.response_type == "transcript":
            raw_response = self._extract_raw_response(model_response) # TODO validar stream
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
            raise ValueError(f"Unsupported message type: `{type(message)}`")
        
        data = encode_data_to_bytes(audio_data)        
        return data

    def _process_message_task(self, message: Message):
        if self.task_multimodal_inputs.data:
            content = self._process_multimodal_inputs(message)
        else:
            raise AttributeError(
                "A message object was passed but neither `multimodal_task_inputs` "
                "were defined"
            )            
        return content        

    def _process_multimodal_inputs(self, message: Message) -> bytes:
        content = None
        audio_path = self.task_multimodal_inputs.data.get("audio", None)

        if audio_path:
            content = self._get_content_from_message(audio_path, message)
            
        if content is None:
            raise ValueError(f"No audio found in paths: `{self.task_inputs.data}`")            
        return content

    def _set_model(self, model: Union[ASRModel, ModelGateway]):
        if model.model_type == "asr":
            self.register_buffer("model", model)
        else:
            raise TypeError(f"`model` need be a `asr` model, given `{type(model)}`")

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
