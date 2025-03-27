import platform
import os
from contextlib import contextmanager
from functools import wraps

import msgspec
from opentelemetry.trace import SpanKind

from msgflow.envs import envs
from msgflow.telemetry.tracer import get_tracer
from msgflow.version import __version__ as msgflow_version


class Spans:
    
    def __init__(self):
        self.tracer = get_tracer()

    def start_span(self, name, attributes=None, kind=SpanKind.INTERNAL):
        """ Base method for start a span. """
        return self.tracer.start_span(name, attributes=attributes, kind=kind)

    def end_span(self, span):
        """ Base method for ending a span. """
        span.end()

    @contextmanager
    def span_context(self, name, attributes=None, kind=SpanKind.INTERNAL):
        """ Generic context manager to create and manage a span. """
        span = self.start_span(name, attributes, kind)
        try:
            yield span
        finally:
            self.end_span(span)

    @contextmanager
    def init_flow(self, module_name, message, encoded_state_dict):
        attributes = {}
        attributes["msgflow.version"] = msgflow_version
        attributes["msgflow.workflow.name"] = module_name
        if message:
            attributes["msgflow.execution_id"] = message.get("execution_id")
            attributes["msgflow.user_id"] = message.get("user_id")
            attributes["msgflow.chat_id"] = message.get("chat_id")        
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
    def init_module(self, module_name):
        attributes = {}
        attributes["msgflow.nn.module.name"] = module_name
        span_name = "Module Initialized"
        with self.span_context(span_name, attributes) as span:
            yield span

    @contextmanager
    def tool_usage(self, tool_callings):
        calls = [{"id": call[0], "name": call[1], "parameters": call[2]} 
                 for call in tool_callings]
        encoded_calls = msgspec.json.encode(calls)
        attributes = {"msgflow.nn.tool.callings": encoded_calls}
        with self.span_context("Tool Usage", attributes) as span:
            yield span

    @contextmanager
    def custom_span(self, name, attributes=None, kind=SpanKind.INTERNAL):
        with self.span_context(name, attributes, kind) as span:
            yield span        

spans = Spans()

def trace(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None, 
):
    def decorator(func):        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with spans.custom_span(name or func.__name__, attributes=attributes) as span:
                return func(*args, **kwargs)
        return wrapper
    return decorator                         

def trace_tool_library_call(forward):
    def wrapper(self, tool_callings):
        with self._spans.tool_usage(tool_callings) as span:
            tool_responses = forward(self, tool_callings)
            if envs.telemetry_capture_tool_call_responses:
                responses = []
                for id, response in tool_responses.items():
                    responses.append({"id": id, "response": response})
                encoded_responses = msgspec.json.encode(responses)
                span.set_attribute("msgflow.nn.tool.responses", encoded_responses)                
            return tool_responses
    return wrapper

def trace_agent_prepare_model_execution(_prepare_model_execution):
    def wrapper(self, model_state):
        attributes = {}
        attributes["msgflow.nn.agent.method.name"] = "_prepare_model_execution"
        with self._spans.custom_span("Prepare Model Execution", attributes) as span:            
            agent_state, system_prompt, tool_schemas = _prepare_model_execution(self, model_state)
            if envs.telemetry_capture_agent_prepare_model_execution:
                encoded_state = msgspec.json.encode(agent_state)
                encoded_tool_schemas = msgspec.json.encode(tool_schemas)                
                span.set_attribute(f"{prefix_span}.agent_state", encoded_state)
                span.set_attribute(f"{prefix_span}.system_prompt", system_prompt or "")
                span.set_attribute(f"{prefix_span}.tool_schemas", encoded_tool_schemas)                              
            return agent_state, system_prompt, tool_schemas
    return wrapper
