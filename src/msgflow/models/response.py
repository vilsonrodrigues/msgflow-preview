from typing import Literal
from gevent.event import Event
from gevent.queue import Queue


class _BaseResponse:

    def set_response_type(self, response_type: str):
        if isinstance(response_type, str):
            self.response_type = response_type
        else:
            raise TypeError("`response_type` requires str"
                            f"given `{type(response_type)}`")


class ModelResponse(_BaseResponse):
    response_type: Literal[
        "audio_embedding",
        "audio_generation",
        "audio_text_generation",
        "image_embedding",        
        "image_generation",       
        "image_text_generation",
        "structured",        
        "tool_call",
        "transcript",
        "translate",
        "text_classification",        
        "text_embedding",
        "text_generation",
    ] = None
    data = None

    def add(self, data):
        self.data = data

    def consume(self):
        return self.data


class ModelStreamResponse(_BaseResponse):
    response_type: Literal[
        "audio_generation", "structured", "text_generation", "tool_call"
    ] = None
    first_chunk_event = Event()
    queue = Queue()

    def add(self, data):
        self.queue.put_nowait(data)

    def consume(self):
        while not self.queue.empty():
            chunk = self.queue.get()
            if chunk is None:
                break
            yield chunk