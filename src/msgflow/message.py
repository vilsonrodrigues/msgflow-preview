from collections import OrderedDict
from uuid import uuid4
from typing import Any, Optional, Union
from msgflow.accessor import Accessor


class _CoreMessage(Accessor):

    def __init__(self, user_id: str, chat_id: str):
        super().__init__()
        self.execution_id = str(uuid4())
        self.user_id = user_id
        self.chat_id = chat_id
        self._route = []
        
    def get_route(self):
        return " -> ".join(self._route)


class Message(_CoreMessage):
    r"""TODO class description"""

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
        user_id: Optional[str] = str(uuid4()),
        chat_id: Optional[str] = str(uuid4()),
    ):
        super().__init__(user_id, chat_id)
        self.content = content
        self.text = text
        self.context = context
        self.audios = audios
        self.images = images
        self.videos = videos
        self.extra = extra
        self.outputs = OrderedDict()
        self.response = OrderedDict()    

    def get_response(self):
        if self.get("response"):
            return next(iter(self.get("response").values()))
        else:
            return self.get("response")

    def __repr__(self):
        to_ignore = ["_route"]
        attrs = [
            (k, v) for k, v in self._attributes.items()
            if k not in to_ignore
        ]
        attrs_str = "\n".join(f"   {k}={repr(v)}" for k, v in attrs)
        return f"{self.__class__.__name__}(\n{attrs_str}\n)"

    def in_msg(self, name: str) -> bool:
        """ Check if data id (name) is in message """
        if name in self._route:
            return True
        else:
            return False
