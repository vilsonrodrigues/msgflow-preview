from copy import deepcopy
from uuid import uuid4
from typing import Any, Dict, Optional, Union
from typing_extensions import Self
from msgflow.dotdict import dotdict


class _CoreMessage(dotdict):

    def __init__(self, user_id: str, user_name: str, chat_id: str):
        super().__init__()
        self.metadata = {
            "execution_id": str(uuid4()),
            "user_id": user_id,
            "user_name": user_name,
            "chat_id": chat_id
        }

    def clone(self) -> Self:
        return deepcopy(self)

class Message(_CoreMessage):

    def __init__(
        self,
        *,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = {},
        texts: Optional[Dict[str, Any]] = {},
        audios: Optional[Dict[str, Any]] = {},
        images: Optional[Dict[str, Any]] = {},
        videos: Optional[Dict[str, Any]] = {},
        extra: Optional[Dict[str, Any]] = {},
        user_id: Optional[str] = str(uuid4()),
        user_name: Optional[str] = None,
        chat_id: Optional[str] = str(uuid4()),
    ):
        super().__init__(user_id, user_name, chat_id)
        self.content = content
        self.texts = texts
        self.context = context
        self.audios = audios
        self.images = images
        self.videos = videos
        self.extra = extra
        self.outputs = {}
        self.response = {}

    def get_response(self):
        if self.get("response"):
            return next(iter(self.get("response").values()))
        else:
            return self.get("response")
