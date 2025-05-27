from queue import Empty, Queue
from threading import Event
from typing import Literal


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
        "moderation",
        "structured",        
        "tool_call",
        "transcript",
        "translate",
        "text_classification",        
        "text_embedding",
        "text_generation",
    ] = None

    def __init__(self):
        self.data = None    

    def add(self, data):
        self.data = data

    def consume(self):
        return self.data


class ModelStreamResponse(_BaseResponse):
    response_type: Literal[
        "audio_generation", "structured", "text_generation", "tool_call"
    ] = None

    def __init__(self):        
        self.first_chunk_event = Event()
        self.queue = Queue()

    def add(self, data):
        """Add data to the stream queue (thread-safe)."""
        self.queue.put(data)

    def consume(self):
        """Generator that yields chunks from the queue until None is received."""
        while True:
            try:
                chunk = self.queue.get(timeout=1.0)  # timeout to avoid infinite blocking
                if chunk is None:
                    break
                yield chunk
            except Empty:
                continue
