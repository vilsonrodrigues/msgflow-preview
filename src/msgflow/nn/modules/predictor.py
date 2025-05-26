from typing import Any, Dict, Optional, Union
from msgflow.message import Message
from msgflow.models.base import BaseModel
from msgflow.models.gateway import ModelGateway
from msgflow.nn.modules.module import Module


class Predictor(Module):
    """Predictor is a generic Module type that uses Classifier, Regressors, Detectors and Segmenters
    to generate insights above data.

    Args:
        name: Predictor name in snake case format.
        model: Predictor Model client.
        task_inputs: Fields of the Message object that will be the input to the task.
        response_mode: What the response should be. Has five options:
            * `plain_response` (default): Returns the transcriber response.
            * `response`: Write on `response` field in Message object.
            * `context`: Write on `context` field in Message object.
               It`s insert how `context.transcriber_name`.
            * `outputs`: Write on `outputs` field in Message object.
               It`s insert how `outputs.transcriber_name`.
    """

    def __init__(
        self,
        name: str,
        model: Union[BaseModel, ModelGateway],
        *,
        task_inputs: Optional[str] = None,
        task_multimodal_inputs: Optional[Dict[str, str]] = None,        
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        execution_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.set_name(name)    
        self._set_model(model)    
        self._set_execution_kwargs(execution_kwargs)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)        
        self._set_task_inputs(task_inputs)
        self._set_task_multimodal_inputs(task_multimodal_inputs)        

    def forward(self, message: Union[Any, Message]):
        model_preference = self.get_model_preference(message)
        data = self._prepare_task(message)
        model_response = self._execute_model(data, model_preference)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(self, data, model_preference=None):
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = self.model(**model_execution_params)
        return model_response

    def _prepare_model_execution(self, data, model_preference=None):
        model_execution_params = self.execution_kwargs or {}
        model_execution_params["data"] = data
        if model_preference:
            model_execution_params["model_preference"] = model_preference        
        return model_execution_params        

    def _process_model_response(self, model_response, message):
        if model_response.response_type == "audio_generation":
            raw_response = self._extract_raw_response(model_response)
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    def _prepare_task(self, message):
        if isinstance(message, Message):
            data = self._process_message_task(message)
        else:
            data = message                
        return data

    def _process_message_task(self, message: Message):
        if self.task_inputs:
            content = self._process_task_inputs(message)
        elif self.task_multimodal_inputs:
            content = self._process_task_multimodal_inputs(message)
        else:
            raise AttributeError(
                "A message object was passed but neither `task_inputs` "
                "nor `multimodal_task_inputs` were defined"
            )            
        return content

    def _process_task_inputs(self, message: Message):
        content = self._get_content_from_message(self.task_inputs, message)
        return content

    def _process_task_multimodal_inputs(self, message: Message):
        content = None

        audio_path = self.task_multimodal_inputs.get("audio", None)
        image_path = self.task_multimodal_inputs.get("image", None)
        file_path = self.task_multimodal_inputs.get("file", None)

        if audio_path:
            content = self._get_content_from_message(audio_path, message)
        elif image_path:
            content = self._get_content_from_message(image_path, message)
        elif file_path:
            content = self._get_content_from_message(file_path, message)

        return content

    def _set_model(self, model: Union[BaseModel, ModelGateway]):
        if isinstance(model, (BaseModel, ModelGateway)):
            self.register_buffer("model", model)
        else:
            raise TypeError(f"`model` need be a `BaseModel` model, given `{type(model)}`")
