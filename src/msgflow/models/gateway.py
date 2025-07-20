from datetime import time, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from msgflow.exceptions import ModelRouterError
from msgflow.logger import logger
from msgflow.models.base import BaseModel
from msgflow.models.model import Model
from msgflow.models.response import ModelResponse, ModelStreamResponse


class ModelGateway:
    """
    Routes calls to a list of supported AI models, with fallback, retries, 
    initial model selection, and timing constraints (configured via HH:MM strings).
    """
    msgflow_type = "model_gateway"
    model_types = None

    def __init__(
        self,
        models: List[BaseModel],
        time_constraints: Optional[Dict[str, List[Tuple[str, str]]]] = None
    ):
        """
        Args:
            models: 
                A list of BaseModel instances (at least 2).
            time_constraints: An optional dictionary mapping model_id to a list of string tuples 
                (start_time, end_time). The listed models will NOT be used if the current time is
                within any of the specified ranges. Strings must be in the format "HH:MM" (e.g. 
                "22:00", "06:00").
                Example: {'model-A': [('22:00', '06:00')]}
                prohibits 'model-A' between 22:00 and 06:00.

        Raises:
            ModelRouterError: 
                Raised when all models fail or are restricted.
            ValueError:
                Raised for misconfiguration in time formats or duplicate model IDs.
            TypeError: 
                Raised for invalid argument types.                
        """        
        self._model_id_to_index: Dict[str, int] = {}
        self.raw_time_constraints = time_constraints
        self._set_models(models)

        try:
            self.parsed_time_constraints = self._parse_time_constraints(time_constraints) if time_constraints else {}
        except ValueError as e:
            logger.error(f"Error to parse time_constraints: {e}")
            raise ValueError(f"Invalid format in time_constraints: {e}") from e

        # Validates if the model_ids in time_constraints exist (uses the keys from the parsed dict)
        for model_id in self.parsed_time_constraints:
            if model_id not in self._model_id_to_index:
                logger.warning(f"The model_id `{model_id}` in time constraints not found in the provided models")

        self.current_model_index = 0
        logger.debug(f"ModelGateway initialized with {len(self.models)} models. Type: `{self.model_type}`. Max fails: `{self.max_retries}`")
        if self.parsed_time_constraints:
            logger.debug(f"Time constraints applied to models: {list(self.parsed_time_constraints.keys())}")

    def _parse_time_constraints(self, constraints: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> Dict[str, List[Tuple[time, time]]]:
        """
        Validates and converts "HH:MM" time strings into datetime.time objects.

        Raises:
            ValueError: If a time string is in an invalid format.
            TypeError: If the constraint data structure is incorrect.
        """
        if constraints is None:
            return {}

        parsed_constraints: Dict[str, List[Tuple[time, time]]] = {}
        time_format = "%H:%M"

        for model_id, intervals in constraints.items():
            if not isinstance(intervals, list):
                raise TypeError(f"Constraints for `{model_id}` must be a list of tuples (start, end). Given: `{type(intervals)}`")
            parsed_intervals = []
            for i, interval in enumerate(intervals):
                if not isinstance(interval, (tuple, list)) or len(interval) != 2: # Tuples or lists
                    raise TypeError(f"Interval #{i+1} for `{model_id}` must be a tuple/list of two strings (start_time_str, end_time_str). Given: `{interval}`")

                start, end = interval
                if not isinstance(start, str) or not isinstance(end, str):
                     raise TypeError(f"Start and end times in range #{i+1} for `{model_id}` must be strings. Given: `({type(start)}, {type(end)})`")

                try:
                    start_t = datetime.strptime(start, time_format).time()
                    end_t = datetime.strptime(end, time_format).time()
                    parsed_intervals.append((start_t, end_t))
                except ValueError as e:
                    raise ValueError(f"Invalid time format in range #{i+1} for `{model_id}`. Use 'HH:MM'. Error parsing `{start}` or `{end}`: {e}") from e

            parsed_constraints[model_id] = parsed_intervals
        return parsed_constraints

    def _is_time_restricted(self, model_id: str) -> bool:
        """Checks if the model is constrained at the current time using the parsed constraints"""
        # Access constraints already converted to `time`
        if model_id not in self.parsed_time_constraints:
            return False

        now = datetime.now().time()

        for start_time, end_time in self.parsed_time_constraints[model_id]:
            if start_time <= end_time:
                if start_time <= now <= end_time:
                    logger.debug(f"Model `{model_id}` restricted. Current time `{now}` is between `{start_time}` and `{end_time}`")
                    return True
            else: # Interval crosses midnight
                if now >= start_time or now <= end_time:
                    logger.debug(f"Restricted model `{model_id}`. Current time `{now}` is in the range crosses midnight: `{start_time} - {end_time}`")
                    return True
        return False

    def _set_models(self, models: List[BaseModel]):
        if not models or not isinstance(models, list):
             raise TypeError("`models` must be a non-empty list of `BaseModel` instances")

        if not all(isinstance(model, BaseModel) for model in models):
            raise TypeError("`models` requires inheriting from `BaseModel`")

        if len(models) < 2:
             logger.warning(f"`models` has only {len(models)} models. Fallback will not be effective")

        model_types = set()
        model_ids = set()
        for i, model in enumerate(models):
            if not hasattr(model, "model_type") or not model.model_type:
                 raise AttributeError(f"Model in {i} position does not have a valid `model_type` attribute")
            if not hasattr(model, "model_id") or not model.model_id:
                 raise AttributeError(f"Model in {i} position  does not have a valid `model_id` attribute")
            if not hasattr(model, "provider"):
                 raise AttributeError(f"Model `{model.model_id}` does not have a valid `provider` attribute")

            model_types.add(model.model_type)
            if model.model_id in model_ids:
                 raise ValueError(f"Duplicate model ID found: `{model.model_id}`. IDs must be unique")
            model_ids.add(model.model_id)
            self._model_id_to_index[model.model_id] = i

        if len(model_types) > 1:
            raise TypeError("All models in `models` must be of the same `model_type`. "
                            f"Given: `{model_types}`")

        self.models = models
        self.model_type = list(model_types)[0]

    def _execute_model(self, model_preference: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Attempts to execute the call on the configured models, respecting
        time constraints and failure limits.
        """
        if not self.models:
            raise ModelRouterError([], [], message="No model configured on gateway")

        available_models = [model for model in self.models if not self._is_time_restricted(model.model_id)]
        
        if not available_models:
            raise ModelRouterError([], [], message="No model available due to time constraints")

        if model_preference:
            preferred_model = next((m for m in available_models if m.model_id == model_preference), None)
            if preferred_model:
                available_models = [preferred_model] + [m for m in available_models if m != preferred_model]

        failures = []
        
        for model in available_models:
            try:
                response = model(**kwargs)
                return response
            except Exception as e:
                logger.debug(f"Model `{model.model_id}` ({model.provider}) failed to execute: {e}", exc_info=False)
                failures.append((model.model_id, model.provider, e))

        error_message = f"All {len(available_models)} available models failed"
        logger.error(error_message)
        raise ModelRouterError(
            [failure[2] for failure in failures], 
            failures, 
            message=error_message
        )

    def __call__(
        self, *, model_preference: Optional[str] = None, **kwargs: Any
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """
        Executes the call on the gateway.

        Args:
            model_preference: 
                The ID of the model that should be tried first.
                If None, starts from the last model used or the first one.
            kwargs: Arguments to pass to the __call__ method of the selected model.

        Returns:
            The response of the first model that executes successfully.

        Raises:
            ModelRouterError: If all models fail consecutively up to the `max_retries` 
                limit, or if no models are available/functional.
        """
        return self._execute_model(model_preference=model_preference, **kwargs)

    async def acall(self, *args, **kwargs):
        """Async interface to `__call__`."""
        return self.__call__(*args, **kwargs)

    def serialize(self) -> Dict[str, Any]:
        """Serializes the gateway state including time constraints as strings."""
        serialized_models = [model.serialize() for model in self.models]
        state = {
            "time_constraints": self.raw_time_constraints,
            "models": serialized_models
        }
        data = {"msgflow_type": self.msgflow_type, "state": state}
        return data

    @classmethod
    def from_serialized(cls, data: Dict[str, Any]) -> "ModelGateway":
        """
        Creates a ModelGateway instance from serialized data.

        Args:
            data: The dictionary of serialized models.
        """        
        if data.get("msgflow_type") != cls.msgflow_type:
             raise ValueError(f"Incorrect msgflow type. Expected `{cls.msgflow_type}`, "
                              f"given `{data.get('msgflow_type')}`")

        state = data.get("state", {})
        serialized_models = state.get("models", [])
        if not serialized_models:
            raise ValueError("Serialized data does not contain templates")

        models = [Model.from_serialized(**m_data) for m_data in serialized_models]
        time_constraints = state.get("time_constraints")

        return cls(models=models, time_constraints=time_constraints)
