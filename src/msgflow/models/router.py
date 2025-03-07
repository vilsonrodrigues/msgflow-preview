from typing import Any, Dict, List, Optional
from msgflow.exceptions import ModelRouterError
from msgflow.models.base import BaseModel


class ModelRouter:
    
    msgflow_type = "model_router"   
    current_model_index = 0
    model_types = None

    def __init__(self, models: List[BaseModel], max_model_failures: Optional[int] = 3):
        self._set_models(models)
        self.max_model_failures = max_model_failures

    def _set_models(self, models):
        if not all(isinstance(model, BaseModel) for model in models):    
            raise TypeError("`models` requires inheriting from `models.base.BaseModel`")
        if len(models) <= 1:
            raise ValueError(f"`models` requires 2 or more models given {len(models)}")
        model_types = set(model.model_type for model in models)
        if len(model_types) != 1:
            raise TypeError("`models` requires that it all be of the same type "
                            f"given {model_types}")
        self.models = models
        self.model_types = model_types[0]

    def _rotate_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.models)

    def _execute_model(self, **kwargs):
        retries = 0
        exceptions = []
        model_info = []

        while retries < self.max_model_failures:
            try:
                response = self.models[self.current_model_index](**kwargs)
                return response
            except Exception as e:
                model_id = self.models[self.current_model_index].model_id
                provider = self.models[self.current_model_index].provider
                exceptions.append(e)
                model_info.append((model_id, provider, e))
                retries += 1
                self._rotate_model()

        raise ModelRouterError(exceptions, model_info)

    def __call__(self, **kwargs):
        response = self._execute_model(**kwargs)
        return response

    def serialize(self) -> Dict[str, Any]:
        serialized_models = [model.serialize() for model in self.models]
        state = {"max_model_failures": self.max_model_failures,
                 "models": serialized_models}
        data = {"msgflow_type": self.msgflow_type,
                "state": state}
        return data
    