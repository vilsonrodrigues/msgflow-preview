import uuid
from collections import OrderedDict
from typing import Any, Optional, Union
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind  
from msgflow.accessor import Accessor


class _CoreMessage(Accessor):
    _route = []
    tracer = trace.get_tracer(__name__)

    def __init__(self, user_id: str, chat_id: str, trace_ctx: Context):
        super().__init__()
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.chat_id = chat_id
        self.trace_ctx = trace_ctx

    def get_route(self):
        return " -> ".join(self._route)

    def trace(self, span_name: str, content: Any):        
        with self.tracer.start_as_current_span(
            span_name,
            context=self.trace_ctx,
            kind=SpanKind.INTERNAL
        ) as span:
            try:
                # TODO: you may need to convert the obj content to str
                span.set_attribute(span_name, content)
                return
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise


class Message(_CoreMessage):
    r"""TODO class description"""

    outputs = OrderedDict()
    response = OrderedDict()

    def __init__(
        self,
        *,
        content: Optional[Union[str, OrderedDict[str, Any]]] = None,
        context: Optional[OrderedDict[str, Any]] = OrderedDict(),
        text: Optional[OrderedDict[str, Any]] = OrderedDict(),
        audios: Optional[OrderedDict[str, Any]] = OrderedDict(),
        images: Optional[OrderedDict[str, Any]] = OrderedDict(),
        videos: Optional[OrderedDict[str, Any]] = OrderedDict(),
        extra: Optional[OrderedDict[str, Any]] = OrderedDict(),
        user_id: Optional[str] = str(uuid.uuid4()),
        chat_id: Optional[str] = str(uuid.uuid4()),
        trace_ctx: Optional[Context] = Context()
    ):
        super().__init__(user_id, chat_id, trace_ctx)
        self.content = content
        self.text = text
        self.context = context
        self.audios = audios
        self.images = images
        self.videos = videos
        self.extra = extra

    def get_response(self):
        if self.get("response"):
            return next(iter(self.get("response").values()))
        else:
            return self.get("response")

    def __repr__(self):
        to_exclude = ["user_id", "chat_id", "trace_ctx", "tracer", "_route"]
        attrs = [
            (k, v) for k, v in self.__dict__.items() 
            if k not in to_exclude
        ]
        attrs_str = "\n".join(f"   {k}={repr(v)}" for k, v in attrs)
        return f"{self.__class__.__name__}(\n{attrs_str}\n)"  

    def in_msg(self, name: str) -> bool:
        """ Check if data id (name) is in message """
        if name in self._route:
            return True
        else:
            return False