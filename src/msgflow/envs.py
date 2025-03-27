import os
from typing import Any, Literal
from msgspec_ext import BaseSettings, SettingsConfigDict


def set_envs(**kwargs: Any):
    """Sets environment variables based on named arguments.

    Args:
        kwargs:
            Named arguments where name is the key of the
            environment variable and value is the value
            to be assigned.

    !!! example
        ```python
        set_envs(VERBOSE=0, LOCAL_RANK=0)
        ```

    """
    for key, value in kwargs.items():
        os.environ[key.upper()] = str(value)


class EnvironmentVariables(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".msgflow_env",
        env_prefix="MSGFLOW_",
    )
    
    # Max objects in cache to functions
    # max_lru_cache: int = 16
    
    # If set to False, msgflow will not print logs
    # If set to True, msgflow will print logs
    verbose: bool = False

    # Logging configuration
    # If set to False, msgflow will not configure logging
    # If set to True, msgflow will configure logging using 
    #    the default configuration or the configuration 
    #    file specified by MSGFLOW_LOGGING_CONFIG_PATH
    configure_logging: bool = True
    logging_config_path: str = None

    # Timeout in secounds to a tool execution
    # default is None, the functions has not Timeout
    tool_timeout: int = None

    # This is used for configuring the default logging level
    logging_level: str = "INFO"

    # if set, MSGFLOW_LOGGING_PREFIX will be prepended to all log messages
    logging_prefix: str = ""
    
    # Trace function calls
    # If set to True, msgflow will trace function calls. Useful for debugging
    trace_function: bool = False

    # if set, msgflow will track executions in nn modules using OTel
    telemetry_requires_trace: bool = False 

    # OTLP endpoint
    telemetry_otlp_endpoint: str = "http://localhost.com:4321"

    # Span exporter type
    telemetry_span_exporter_type: Literal["console", "otlp"] = "console"

    # Capture state dict
    telemetry_capture_state_dict: bool = False

    # Capture platform details
    telemetry_capture_platform: bool = False

    # Capture tool call responses
    telemetry_capture_tool_call_responses: bool = True

    # Capture agent state, system prompt and tool schemas
    telemetry_capture_agent_prepare_model_execution: bool = False

    # State checkpoint, if True, if a module output is in message, skip process
    # if False, reprocess
    state_checkpoint: bool = False

    # Max attemps to model clients
    model_stop_after_attempt: int = 5

    # Model retry delay
    model_stop_after_delay: int = 0

    # Max attemps to tool call
    tool_stop_after_attempt: int = 5

    # Tool retry delay
    tool_stop_after_delay: int = 0

envs = EnvironmentVariables()
