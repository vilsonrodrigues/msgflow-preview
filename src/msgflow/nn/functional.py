# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
from typing import Callable, List, Literal, Optional, Tuple
import gevent
from msgflow.message import Message


def bcast_gather(
    message: Message,
    to_send: List[Callable],
    response_mode: Optional[str] = "outputs",
) -> Message:
    """ Brodcast 1 message to N modules and gather """
    if not isinstance(message, Message):
        raise TypeError("`message` requires be a `msgflow.Message`")
    
    if not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` requires be a list of Callable")
    
    if not isinstance(response_mode, str):
        raise TypeError("`response_mode` requires be a string")
    else:
        if response_mode == "":
            raise ValueError("`response_mode` requires be a string not empty")

    tasks = [gevent.spawn(module, message) for module in to_send]
    results = gevent.joinall(tasks)

    for module, response in zip(to_send, results):
        message.set(f"{response_mode}.{module.name}", response.value)

    return message


def scatter_gather(
    messages: List[Message],
    to_send: List[Callable],
    response_mode: Optional[Literal["context", "outputs"]] = "outputs",
) -> Tuple[Message]:
    """Scatter N messages to N modules"""
    if not all(isinstance(message, Message) for message in messages):
        raise TypeError("`messages` requires be a list of `msgflow.Message`")
    
    if not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` requires be a list of Callable")
        
    if not isinstance(response_mode, str):
        raise TypeError("`response_mode` requires be a string")
    else:
        if response_mode == "":
            raise ValueError("`response_mode` requires be a string not empty")

    tasks = [gevent.spawn(module, msg) for module, msg in zip(to_send, messages)]
    results = gevent.joinall(tasks)

    for module, message, response in zip(to_send, messages, results):
        message.set(f"{response_mode}.{module.name}", response.value)

    return tuple(messages)