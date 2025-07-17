import platform
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec
from opentelemetry.trace import SpanKind, Status, StatusCode

from msgflow.envs import envs
from msgflow.message import Message
from msgflow.telemetry.tracer import get_tracer
from msgflow.version import __version__ as msgflow_version


class Spans:
    
    def __init__(self):
        self.tracer = get_tracer()

    @contextmanager
    def span_context(self, name: str, attributes: Dict[str, Any] = None, kind: str = SpanKind.INTERNAL):
        """Generic context manager to create and manage a span."""
        with self.tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    @contextmanager
    def init_flow(
        self, 
        module_name: str, 
        message: Optional[Message] = None, # TODO: pass metadata directly
        encoded_state_dict: Optional[str] =  None
    ):
        attributes = {}
        attributes["msgflow.version"] = msgflow_version
        attributes["msgflow.workflow.name"] = module_name
        if message:
            attributes["msgflow.metadata"] = message.get("metadata")
        if encoded_state_dict:
            attributes["msgflow.state_dict"] = encoded_state_dict
        if envs.telemetry_capture_platform:            
            attributes["platform"] = platform.platform()
            attributes["platform.version"] = platform.version()
            attributes["platform.python.version"] = platform.python_version()
            attributes["platform.num_cpus"] = os.cpu_count()
        
        span_name = "Workflow Initialized"
        with self.span_context(span_name, attributes) as span:
            yield span

    @contextmanager
    def init_module(self, module_name: str):
        attributes = {}
        attributes["msgflow.nn.module.name"] = module_name
        span_name = "Module Initialized"
        with self.span_context(span_name, attributes) as span:
            yield span

    @contextmanager
    def tool_usage(self, tool_callings: List[Tuple[str, str, Any]]):
        calls = [{"id": call[0], "name": call[1], "parameters": call[2]} 
                 for call in tool_callings]
        encoded_calls = msgspec.json.encode(calls)
        attributes = {"msgflow.nn.tool.callings": encoded_calls}
        with self.span_context("Tool Usage", attributes) as span:
            yield span

spans = Spans()

def instrument(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None, 
) -> Callable:
    """
    Decorator that instruments a function with a tracing span.

    This decorator creates a span using the provided name and attributes
    when the decorated function is called. If an exception occurs during
    execution, the exception is recorded in the span and the span status is
    set to error.

    Args:
        name:
            The name of the span. If not provided, the function's name is used.
        attributes:
            A dict of attributes to attach to the span.
            Useful for adding metadata for tracing purposes.

    Returns:
        A decorated function that is wrapped with span instrumentation.

    Raises:
        Exception: Reraises any exception thrown by the wrapped function after
            recording it in the span.
    """    
    def decorator(func):        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with spans.span_context(name or func.__name__, attributes=attributes) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise            
        return wrapper
    return decorator                    

def instrument_tool_library_call(forward):
    def wrapper(
        self, 
        tool_callings: List[Tuple[str, str, Any]],
        model_state: Optional[List[Dict[str, Any]]] = None,
        injected_kwargs: Optional[Dict[str, Any]] = {}
    ):
        with self._spans.tool_usage(tool_callings) as span:
            tool_execution_result = forward(self, tool_callings, model_state, injected_kwargs)
            if envs.telemetry_capture_tool_call_responses:
                span.set_attribute("msgflow.nn.tool.responses", tool_execution_result.to_json())
            return tool_execution_result
    return wrapper

def instrument_agent_prepare_model_execution(_prepare_model_execution):
    def wrapper(self, *args, **kwargs):
        prefix_span = "msgflow.nn.Agent"
        attributes = {}
        attributes[f"{prefix_span}.method.name"] = "_prepare_model_execution"
        with self._spans.span_context("Prepare Model Execution", attributes) as span:            
            model_execution_params = _prepare_model_execution(self, *args, **kwargs)
            if envs.telemetry_capture_agent_prepare_model_execution:
                encoded_state = msgspec.json.encode(model_execution_params["messages"])
                encoded_tool_schemas = msgspec.json.encode(model_execution_params["tool_schemas"])
                system_prompt = model_execution_params["system_prompt"] or ""
                span.set_attribute(f"{prefix_span}.agent_state", encoded_state)
                span.set_attribute(f"{prefix_span}.system_prompt", system_prompt)
                span.set_attribute(f"{prefix_span}.tool_schemas", encoded_tool_schemas)                              
            return model_execution_params
    return wrapper
