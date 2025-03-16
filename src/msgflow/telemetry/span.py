#
#
#https://github.com/traceloop/openllmetry/tree/main/packages
#tracer = get_tracer()

# Criar um span
#with tracer.start_as_current_span("exemplo_operacao") as span:
#    # Adicionar alguns atributos ao span
#    span.set_attribute("exemplo.atributo", "valor")
    def trace(self, key: str, value: Any, span_name: Optional[str] = None):        
        with self.tracer.start_as_current_span(
            span_name,
            context=self.get("trace_ctx"),
            kind=SpanKind.INTERNAL
        ) as span:
            try:
                span.set_attribute(key, value)
                span.set_status(Status(StatusCode.OK))
                return
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise

from opentelemetry import trace
from msgflow.telemetry.tracer import get_tracer


tracer = get_tracer()

class MsgflowSpans:

    def tool_call(self, ):
        current_span = trace.get_current_span()
        with tracer.start_as_current_span(f"submodule.{self.__class__.__name__}") as span:
            ...
        