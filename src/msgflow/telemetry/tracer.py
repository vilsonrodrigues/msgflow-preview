import os
from typing import Optional, Literal
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, NoOpTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.trace import Span, Status, StatusCode
from msgflow.envs import envs
from msgflow.logger import logger


_tracer_configured = False

def configure_tracer():
    """ Configure OpenTelemetry tracer """
    global _tracer_configured
    
    if _tracer_configured:
        return

    if not envs.requires_trace:
        logger.debug("Tracing disabled, configuring NoOp tracer")
        no_op_provider = NoOpTracerProvider()
        trace.set_tracer_provider(no_op_provider)
        _tracer_configured = True
        return        
        
    attributes = {SERVICE_NAME: "msgflow-telemetry"}    
    resource = Resource.create(attributes)    
    provider = TracerProvider(resource=resource)

    if envs.telemetry_span_exporter_type.lower() == "otlp":
        otlp_exporter = OTLPSpanExporter(endpoint=envs.telemetry_otlp_endpoint)
        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)
        logger.debug(f"Configured OTLP exporter with endpoint: {envs.telemetry_otlp_endpoint}")
    elif envs.telemetry_span_exporter_type.lower() == "console":
        console_exporter = ConsoleSpanExporter()
        processor = BatchSpanProcessor(console_exporter)
        provider.add_span_processor(processor)
        logger.debug("Configured Console exporter")
    
    trace.set_tracer_provider(provider)  
    _tracer_configured = True    

def get_tracer():
    if not _tracer_configured:
        configure_tracer()
    return trace.get_tracer("msgflow.telemetry")
