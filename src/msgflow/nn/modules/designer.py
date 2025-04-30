from typing import Any, Callable, Dict, Literal, Optional, Union
from msgflow.message import Message
from msgflow.models.base import BaseModel
from msgflow.models.gateway import ModelGateway
from msgflow.models.types import (
    ImageTextTo3DModel,
    ImageTextToImageModel,
    ImageTo3DModel,
    ImageToImageModel,
    TextTo3DModel,
    TextToImageModel,
    TextToVideoModel,
    #VideoTextToVideoModel
)
from msgflow.nn.modules.module import Module
from msgflow.utils.encode import encode_data_to_base64


VISION_GEN_MODEL_TYPES = [
    ModelGateway,
    ImageTextTo3DModel,
    ImageTextToImageModel,
    ImageTo3DModel,
    ImageToImageModel,
    TextTo3DModel,
    TextToImageModel,
    TextToVideoModel,
    #VideoTextToVideoModel    
]


class Designer(Module):
    """Designer is a Module type that uses Vision generative models to create content.

    Args:
        name: Designer name in snake case format.
        model: Designer Model client.
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
        model: VISION_GEN_MODEL_TYPES,
        *,
        input_guardrail: Optional[Callable] = None,
        output_guardrail: Optional[Callable] = None,
        task_inputs: Optional[str] = None,
        task_multimodal_inputs: Optional[Dict[str, str]] = None,
        response_format: Optional[Literal["base64", "url"]] = None,
        response_mode: Optional[str] = "plain_response",
        negative_prompt: Optional[str] = None,
        fps: Optional[int] = None,
        duration_seconds: Optional[int] = None,
        aspect_ratio: Optional[str] = None, # TODO
        n: Optional[int] = None, # TODO
        execution_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.set_name(name)
        self._set_duration_seconds(duration_seconds)
        self._set_execution_kwargs(execution_kwargs)
        self._set_fps(fps)
        self._set_input_guardrail(input_guardrail)
        self._set_output_guardrail(output_guardrail)
        self._set_model(model)
        self._set_negative_prompt(negative_prompt)
        self._set_response_mode(response_mode)
        self._set_response_format(response_format)        
        self._set_task_inputs(task_inputs)
        self._set_task_multimodal_inputs(task_multimodal_inputs)        

    def forward(self, message: Union[str, Message]):
        model_preference = self.get_model_preference(message)
        params = self._prepare_task(message)
        model_response = self._execute_model(params, model_preference)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(self, params, model_preference=None):
        model_execution_params = self._prepare_model_execution(params, model_preference)
        if self.attr_is_valid(self.guardrail):
            self._execute_guardrail(model_execution_params)
        model_response = self.model.data(**model_execution_params)
        return model_response

    def _prepare_model_execution(self, params, model_preference=None):        
        model_execution_params = self.execution_kwargs.data or {}
        model_execution_params.update(params)
        if self.negative_prompt.data:
            model_execution_params["negative_prompt"] = self.negative_prompt.data
        if self.fps.data:
            model_execution_params["fps"] = self.fps.data
        if self.duration_seconds.data:
            model_execution_params["duration_seconds"] = self.duration_seconds.data
        if model_preference:
            model_execution_params["model_preference"] = model_preference
        return model_execution_params

    def _prepare_guardrail_execution(self, model_execution_params):
        prompt = model_execution_params.get("prompt")
        image = model_execution_params.get("image", None)
        if image is not None:
            messages = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
            ]
            data = {"data": messages}
        else:
            data = {"data": prompt}
        guardrail_params = data
        return guardrail_params

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
        if isinstance(message, str):
            params = {"prompt": message}
        elif isinstance(message, Message):
            params = self._process_message_task(message)             
        return params

    def _process_message_task(self, message: Message):
        params = {}
        if self.task_inputs.data:
            params["prompt"] = self._process_inputs(message)
        elif self.task_multimodal_inputs.data:
            params.update(self._process_multimodal_inputs(message))
        else:
            raise AttributeError(
                "A message object was passed but neither `task_inputs` "
                "nor `multimodal_task_inputs` were defined"
            )

        if not params:
            raise ValueError("No content found in message object")
        
        return params

    def _process_inputs(self, message: Message):
        content = self._get_content_from_message(self.task_inputs.data, message)
        return content

    def _process_multimodal_inputs(self, message: Message):
        params = {}
        
        image_path = self.task_multimodal_inputs.data.get("image", None)
        mask_path = self.task_multimodal_inputs.data.get("mask", None)

        if image_path:
            image = self._get_content_from_message(image_path, message)
            image_base64 = encode_data_to_base64(image)
            params["image"] = image_base64
        if mask_path:
            mask = self._get_content_from_message(mask_path, message)
            mask_base64 = encode_data_to_base64(mask)
            params["mask"] = mask_base64

        return params

    def _set_model(self, model: Union[BaseModel, ModelGateway]):
        if isinstance(model, tuple(VISION_GEN_MODEL_TYPES)):
            self.register_buffer("model", model)
        else:
            raise TypeError(f"`model` need be a `{str(VISION_GEN_MODEL_TYPES)}` "
                            f"model, given `{type(model)}`")

    def _set_response_format(self, response_format: Optional[str] = None):
        if isinstance(response_format, str) or response_format is None:
            self.register_buffer("response_format", response_format)
        else:
            raise TypeError(f"`response_format` need be a str or given `{type(response_format)}")               

    def _set_negative_prompt(self, negative_prompt: Optional[str] = None):
        if isinstance(negative_prompt, str) or negative_prompt is None:
            self.register_buffer("negative_prompt", negative_prompt)
        else:
            raise TypeError(f"`negative_prompt` need be a str or None given `{type(negative_prompt)}`")

    def _set_fps(self, fps: Optional[int] = None):
        if isinstance(fps, int) or fps is None:
            self.register_buffer("fps", fps)
        else:
            raise TypeError(f"`fps` need be an int or None given `{type(fps)}`")

    def _set_duration_seconds(self, duration_seconds: Optional[int] = None):
        if isinstance(duration_seconds, int) or duration_seconds is None:
            self.register_buffer("duration_seconds", duration_seconds)
        else:
            raise TypeError(f"`duration_seconds` need be an int or None given `{type(duration_seconds)}`")        
  