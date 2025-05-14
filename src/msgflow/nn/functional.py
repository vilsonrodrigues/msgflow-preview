# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
from typing import Any, Callable, List, Optional, Tuple
import gevent
from gevent import Greenlet
from msgflow.logger import logger
from msgflow.message import Message
from msgflow.nn.modules.module import get_callable_name
from msgflow.telemetry.span import trace


@trace("msgflow.nn.F.bcast_gather")
def bcast_gather(
    message: Any,
    to_send: List[Callable],
    timeout: Optional[float] = None,
) -> Tuple[Any]:
    """
    Broadcasts a single message to multiple modules and gathers the responses.

    The given message is sent to each callable in `to_send`, and the responses are collected.
    Then a tuple containing the response of each callable will be returned. If an error occurs, 
    the response of that callable will be None. Includes exception handling and optional timeout.

    If the value of message is None the module will be called without passing parameters.  

    Args:
        message: Any data object to broadcast.
        to_send: List of callable objects (e.g. functions or `Module` instances).
        timeout: Maximum time (in seconds) to wait for responses.

    Returns:
        Tuple containing the responses.

    Raises:
        TypeError: If `to_send` is not a list of callables.
    """
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    tasks: List[Greenlet] = [
        gevent.spawn(module, message) if message is not None else gevent.spawn(module)
        for module in to_send
    ]
    gevent.joinall(tasks, timeout=timeout)
    responses = []
    for task in tasks:
        try:        
            if task.successful():
                responses.append(task.value)
            else:
                responses.append(None)
        except Exception as e:
            logger.error(str(e))                
    responses = tuple(responses)
    return responses


@trace("msgflow.nn.F.msg_bcast_gather")
def msg_bcast_gather(
    message: Message,
    to_send: List[Callable],
    response_mode: Optional[str] = "outputs",
    timeout: Optional[float] = None,
) -> Message:
    """
    Broadcasts a single message to multiple modules and gathers the responses.

    The given message is sent to each callable in `to_send`, and the responses are collected
    in the field specified by `response_mode`, using the module name as the key. Includes
    exception handling and optional timeout to prevent crashes.

    Args:
        message: Instance of `msgflow.Message` to broadcast.
        to_send: List of callable objects (e.g. functions or `Module` instances).
        response_mode: Field in the message where the responses will be stored (default: "outputs").
        timeout: Maximum time (in seconds) to wait for responses (optional).

    Returns:
        Message: The original message with the module responses added.

    Raises:
        TypeError: If `message` is not an instance of `Message`, `to_send` is not a list
            of callables, or `response_mode` is not a string.
        ValueError: If `response_mode` is an empty string or `to_send` is empty.
    """
    if not isinstance(message, Message):
        raise TypeError("`message` must be an instance of `msgflow.Message`")
    
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")
        
    if not isinstance(response_mode, str):
        raise TypeError(f"`response_mode` must be a string, but it was received `{type(response_mode)}`")
    if response_mode == "":
        raise ValueError("`response_mode` cannot be an empty string")
    
    tasks: List[Greenlet] = [gevent.spawn(module, message) for module in to_send]    
    gevent.joinall(tasks, timeout=timeout)
    for module, task in zip(to_send, tasks):
        module_name = get_callable_name(module)
        try:
            if task.successful():
                message.set(f"{response_mode}.{module_name}", task.value)
            else:
                message.set(f"{response_mode}.{module_name}", None)
        except Exception as e:
            message.set(f"{response_mode}.{module_name}", None)
            logger.error(str(e))
    return message


@trace("msgflow.nn.F.scatter_gather")
def scatter_gather(
    messages: List[Any],
    to_send: List[Callable],
    timeout: Optional[float] = None,
) -> Tuple[Any]:
    """
    Scatter a list of messages to a list of modules and gather the responses.

    Each message in `messages` is sent to the corresponding callable in `to_send`, 
    and the responses are collected. Then a tuple containing the response of each 
    callable will be returned. If an error occurs, the response of that callable 
    will be None.

    If the value of message is None the module will be called without passing parameters.    

    Args:
        messages: List of any data object to be distributed.
        to_send: List of callable objects (e.g. functions or `Module` instances).
        timeout: Maximum time (in seconds) to wait for responses.

    Returns:
        Tuple containing the responses.

    Raises:
        TypeError: If `to_send` is not a list of callables.
        ValueError: If the lengths of `messages` and `to_send` do not match.
    """
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")
    
    if len(messages) != len(to_send):
        raise ValueError(f"The size of `messages` ({len(messages)}) "
                         f"must be equal to that of `to_send`: ({len(to_send)})")

    tasks: List[Greenlet] = [
        gevent.spawn(module, message) if message is not None else gevent.spawn(module)
        for module, message in zip(to_send, messages)
    ]
    gevent.joinall(tasks, timeout=timeout)
    responses = []
    for task in tasks:
        try:        
            if task.successful():
                responses.append(task.value)
            else:
                responses.append(None)
        except Exception as e:
            logger.error(str(e))              
    responses = tuple(responses)
    return responses


@trace("msgflow.nn.F.msg_scatter_gather")
def msg_scatter_gather(
    messages: List[Message],
    to_send: List[Callable],
    response_mode: Optional[str] = "outputs",
    timeout: Optional[float] = None,
) -> Tuple[Message]:
    """
    Scatter a list of messages to a list of modules and gather the responses.

    Each message in `messages` is sent to the corresponding callable in `to_send`, 
    and the responses are stored in the field specified by `response_mode` in the 
    respective message. Includes exception handling and optional timeout.

    Args:
        messages: List of `msgflow.Message` instances to be distributed.
        to_send: List of callable objects (e.g. functions or `Module` instances).
        response_mode: Field where the responses will be stored (default: "outputs").
        timeout: Maximum time (in seconds) to wait for responses (optional).

    Returns:
        Tuple[Message]: Tuple containing the messages updated with the responses.

    Raises:
        TypeError: If `messages` is not a list of `Message`, `to_send` is not a list
            of callables, or `response_mode` is not a string.
        ValueError: If `response_mode` is an empty string, or the lengths of `messages`
            and `to_send` do not match.
    """
    # Validações
    if not messages or not all(isinstance(msg, Message) for msg in messages):
        raise TypeError("`messages` must be a non-empty list of `msgflow.Message` instances")
    
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")
    
    if len(messages) != len(to_send):
        raise ValueError(f"The size of `messages` ({len(messages)}) "
                         f"must be equal to that of `to_send`: ({len(to_send)})")
             
    if not isinstance(response_mode, str):
        raise TypeError(f"`response_mode` must be a string, but it was received `{type(response_mode)}`")
    if response_mode == "":
        raise ValueError("`response_mode` cannot be an empty string")

    tasks: List[Greenlet] = [gevent.spawn(module, msg) for module, msg in zip(to_send, messages)]    
    gevent.joinall(tasks, timeout=timeout)
    for module, message, task in zip(to_send, messages, tasks):
        module_name = get_callable_name(module)
        try:
            if task.successful():
                message.set(f"{response_mode}.{module_name}", task.value)
            else:
                message.set(f"{response_mode}.{module_name}", None)
                logger.error(f"{module_name}: {task.exception}")
        except Exception as e:
            message.set(f"{response_mode}.{module_name}", None)
            logger.error(str(e))
    return tuple(messages)
