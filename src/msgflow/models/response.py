import asyncio
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
        self.first_chunk_event = asyncio.Event()
        self.queue = asyncio.Queue()

    async def add(self, data):
        """Add data to the stream queue (async)."""
        await self.queue.put_nowait(data)

    async def consume(self):
        """Async generator that yields chunks from the queue until None is received."""
        while True:
            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                continue