from typing import Dict, Literal
from gevent.event import Event
from gevent.queue import Queue


class _BaseResponse:    
    metadata: Dict = None

    def set_response_type(self, response_type: str):
        if isinstance(response_type, str):
            self.response_type = response_type
        else:
            raise TypeError("`Response` classes requires `response_type` str"
                            f"given {type(response_type)}")

    def set_metadata(self, metadata: Dict):
        if isinstance(metadata, Dict):        
            self.metadata = metadata
        else:
            raise TypeError("`Response` classes requires `metadata` dict"
                            f"given {type(metadata)}") 

class Response(_BaseResponse):
    response_type: Literal[
        "audio_embeddings",
        "audio_generation",
        "audio_text_generation",
        "image_embeddings",        
        "image_generation",        
        "structured",        
        "tool_call",
        "transcript",
        "translate",
        "text_classification",        
        "text_embeddings",
        "text_generation",
    ] = None
    data = None

    def add(self, data):
        self.data = data

    def consume(self):
        return self.data


class StreamResponse(_BaseResponse):
    response_type: Literal[
        "audio_generation" "text_generation", "tool_call", "structured"
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