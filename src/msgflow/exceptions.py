# Temporary
class KeyExhaustedError(Exception):
    """Exception raised when all API keys have been tried and failed."""
    pass

class ToolCallTimeOutError(Exception):
    ...

class ModelRouterError(Exception):
    def __init__(self, exceptions, model_info):
        self.exceptions = exceptions
        self.model_info = model_info
        message = "All model calls failed. Details:\n"
        for i, (model_id, provider, exc) in enumerate(model_info):
            message += f"Model {i + 1}: ID={model_id}, Provider={provider}, Error={str(exc)}\n"
        super().__init__(message)