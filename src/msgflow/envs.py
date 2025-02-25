import os
from typing import Any, Optional
from msgspec_ext import BaseSettings, SettingsConfigDict
from msgspec_ext import FilePath

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


def get_env(key: str, default: Optional[str] = None) -> str:
    """Returns a environment variable value.

    Args:
        key:
            The name of the environment variable to read.
        default:
            The default value to return if the variable is not set.
            The default is None.

    Returns:
        The value of the environment variable or the default value
        if it is not set.

    Example:
        api_key = get_env("API_TOKEN", default="default-token")
    """
    return os.getenv(key, default)


class EnvironmentVariables(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".msgflow_env",
        env_prefix="MSGFLOW_",
    )
    
    # Max objects in cache to functions
    max_lru_cache: int = 16
    
    # If set to False, msgflow will not print logs
    # If set to True, msgflow will print logs
    verbose: bool = False

    # Logging configuration
    # If set to 0, msgflow will not configure logging
    # If set to 1, msgflow will configure logging using 
    #    the default configuration or the configuration 
    #    file specified by MSGFLOW_LOGGING_CONFIG_PATH
    configure_logging: bool = True
    logging_config_path: FilePath

    # Timeout in secounds to a tool execution
    # default is None, the functions has not Timeout
    tool_timeout: int

    # This is used for configuring the default logging level
    logging_level: str = "INFO"

    # if set, MSGFLOW_LOGGING_PREFIX will be prepended to all log messages
    logging_prefix: str = ""
    
    # Trace function calls
    # If set to 1, msgflow will trace function calls. Useful for debugging
    trace_function: bool = False

envs = EnvironmentVariables()
