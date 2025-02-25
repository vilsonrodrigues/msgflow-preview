from typing import Any, Dict, List, Optional
from msgflow.models.base import BaseModel


class ModelRouter:
    
    msgflow_type = "model_router"   
    current_model_index: int = 0

    def __init__(self, models: List[BaseModel], max_model_failures: Optional[int] = 3):
        self._set_models(models)
        self.max_model_failures = max_model_failures

    def _set_models(self, models):
        if not all(isinstance(model, BaseModel) for model in models):    
            raise TypeError("`models` requires inheriting from `BaseModel`")
        if len(models) <= 1:
            raise ValueError(f"`models` requires 2 or more models given {len(models)}")
        model_types = set(model.model_type for model in models)
        if len(model_types) != 1:
            raise TypeError("`models` requires that it all be of the same type "
                            f"given {model_types}")
        self.models = models

    def _rotate_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.models)

    def _execute_model(self, **kwargs):
        retries = 0
        exceptions = []
        while retries < self.max_model_failures:
            try:
                response = self.models[self.current_model_index](**kwargs)
                return response
            except Exception as e:
                exceptions.append(e)
                retries += 1
                self._rotate_model()
        raise Exception("All models failed")

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
    