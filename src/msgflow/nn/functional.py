# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
import asyncio
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple
from msgflow._private.executor import Executor
from msgflow.logger import logger
from msgflow.message import Message
from msgflow.nn.modules.module import get_callable_name
from msgflow.telemetry.span import trace


@trace("msgflow.nn.F.map_gather")
def map_gather(
    to_send: Callable,
    *,
    args_list: List[Tuple[Any, ...]],
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    timeout: Optional[float] = None,
) -> Tuple[Any, ...]:
    """
    Applies the `to_send` function to each set of arguments in `args_list` 
    and `kwargs_list` using Executor and collects the results.

    Args:
        to_send: 
            The callable function to be applied.
        args_list:
            Each tuple contains the positional argumentsvfor the corresponding callable 
            in `to_send`. If `None`, no positional arguments are passed unless specified 
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable 
            in `to_send`. If `None`, no named arguments are passed unless specified 
            individually by an item in `args_list`.
        timeout:
            Maximum time (in seconds) to wait for responses.
    Returns:
        A tuple containing the results of each call to the `f` function. If a call
        fails or times out, the corresponding result will be `None`.

    Raises:
        TypeError: 
            If `f` is not callable.
        ValueError: 
            If `args_list` is not a non-empty list or if `kwargs_list`
            (if provided) is not the same length as `args_list`.

    Examples:
        def add(x, y):
            return x + y
        results = map_function(add, [(1, 2), (3, 4), (5, 6)])
        print(results)  # (3, 7, 11)

        def multiply(x, y=2):
            return x * y
        results = map_function(multiply, [(1,), (3,), (5,)], kwargs_list=[{'y': 3}, {'y': 4}, {'y': 5}])
        print(results)  # (3, 12, 25)

        results = map_function(multiply, [(1,), (3,), (5,)])  # Usa y=2 por default
        print(results)  # (2, 6, 10)
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    if not isinstance(args_list, list) or len(args_list) == 0:
        raise ValueError("`args_list` must be a non-empty list")

    if kwargs_list is not None:
        if not isinstance(kwargs_list, list) or len(kwargs_list) != len(args_list):
            raise ValueError("`kwargs_list` must be a list with the same length as `args_list`")

    executor = Executor.get_instance()
    futures = []

    for i in range(len(args_list)):
        args = args_list[i]
        kwargs = kwargs_list[i] if kwargs_list else {}
        futures.append(executor.submit(to_send, *args, **kwargs))

    done, _ = concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for future in done:
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(None)
    return tuple(responses)


@trace("msgflow.nn.F.scatter_gather")
def scatter_gather(
    to_send: List[Callable],
    args_list: Optional[List[Tuple[Any, ...]]] = None,
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    *,
    timeout: Optional[float] = None,
) -> Tuple[Any, ...]:
    """
    Sends different sets of arguments/kwargs to a list of modules (callables)
    and collects the responses.

    Each callable in `to_send` receives the positional arguments of the corresponding `tuple`
    in `args_list` and the named arguments of the corresponding `dict` in `kwargs_list`.
    If `args_list` or `kwargs_list` are not provided (or are `None`), the corresponding callables
    will be called without positional or named arguments, respectively,
    unless an empty list (`[]`) or empty tuple (`()`) is provided for a specific item.

    Args:
        to_send: 
            List of callable objects (e.g. functions or `Module` instances).
        args_list:
            Each tuple contains the positional argumentsvfor the corresponding callable 
            in `to_send`. If `None`, no positional arguments are passed unless specified 
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable 
            in `to_send`. If `None`, no named arguments are passed unless specified 
            individually by an item in `args_list`.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        Tuple containing the responses for each callable. If an error or timeout occurs for a 
        specific callable, its corresponding response in the tuple will be `None`.

    Raises:
        TypeError: If `to_send` is not a callable list.
        ValueError: If the lengths of `args_list` (if provided) or `kwargs_list`
            (if provided) do not match the length of `to_send`.

    Examples:
        def add(x, y): return x + y
        def multiply(x, y=2): return x * y
        callables = [add, multiply, add]

        # Example 1: Using only args_list
        args = [ (1, 2), (3,), (10, 20) ] # multiply will use its default y
        results = scatter_gather(callables, args_list=args)
        print(results) # (3, 6, 30)

        # Example 2: Using args_list e kwargs_list
        args = [ (1,), (), (10,) ]
        kwargs = [ {'y': 2}, {'x': 3, 'y': 3}, {'y': 20} ]
        results = scatter_gather(callables, args_list=args, kwargs_list=kwargs)
        print(results) # (3, 9, 30)

        # Example 3: Using only kwargs_list (useful if functions have defaults or don't need positional args)
        def greet(name="World"): return f"Hello, {name}"
        def farewell(person_name): return f"Goodbye, {person_name}"
        funcs = [greet, greet, farewell]
        kwargs_for_funcs = [ {}, {'name': "Earth"}, {'person_name': "Commander"} ]
        results = scatter_gather(funcs, kwargs_list=kwargs_for_funcs)
        print(results) # ("Hello, World", "Hello, Earth", "Goodbye, Commander")
    """
    if not isinstance(to_send, list) or not all(callable(f) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    executor = Executor.get_instance()
    futures = []
    for i, f in enumerate(to_send):
        args = args_list[i] if args_list and i < len(args_list) else ()
        kwargs = kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {}
        futures.append(executor.submit(f, *args, **kwargs))
    
    done, _ = concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for FUTURE in done:
        try:
            responses.append(FUTURE.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(None)
    return tuple(responses)


@trace("msgflow.nn.F.msg_scatter_gather")
def msg_scatter_gather(
    to_send: List[Callable],
    messages: List[Message],
    *,
    response_mode: Optional[str] = "outputs",    
    timeout: Optional[float] = None,
) -> Tuple[Message, ...]:
    """
    Scatter a list of messages to a list of modules and gather the responses.

    Each message in `messages` is sent to the corresponding callable in `to_send`, 
    and the responses are stored in the field specified by `response_mode` in the 
    respective message. Includes exception handling and optional timeout.

    Args:
        to_send: List of callable objects (e.g. functions or `Module` instances).    
        messages: List of `msgflow.Message` instances to be distributed.
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
    if not messages or not all(isinstance(msg, Message) for msg in messages):
        raise TypeError("`messages` must be a non-empty list of `msgflow.Message` instances")

    if not to_send or not all(isinstance(f, Callable) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    if len(messages) != len(to_send):
        raise ValueError(f"The size of `messages` ({len(messages)}) "
                        f"must be equal to that of `to_send`: ({len(to_send)})")

    if not isinstance(response_mode, str):
        raise TypeError(f"`response_mode` must be a string, but it was received `{type(response_mode)}`")
    if not response_mode:
        raise ValueError("`response_mode` cannot be an empty string")

    executor = Executor.get_instance()
    futures = [
        executor.submit(f, msg)
        for f, msg in zip(to_send, messages)
    ]
    
    done, _ = concurrent.futures.wait(futures, timeout=timeout)
    for f, message, FUTURE in zip(to_send, messages, done):
        f_name = get_callable_name(f)
        try:
            message.set(f"{response_mode}.{f_name}", FUTURE.result())
        except Exception as e:
            logger.error(f"Error in scattered task for `{f_name}`: {e}")
            message.set(f"{response_mode}.{f_name}", None)
    return tuple(messages)


@trace("msgflow.nn.F.bcast_gather")
def bcast_gather(
    to_send: List[Callable],
    *args,
    timeout: Optional[float] = None,
    **kwargs
) -> Tuple[Any, ...]:
    """
    Broadcasts arguments to multiple callables and gathers the responses.

    Args:
        to_send: List of callable objects (e.g. functions or `Module` instances).    
        *args: Positional arguments.
        timeout: Maximum time (in seconds) to wait for responses.
        **kwargs: Named arguments.

    Returns:
        Tuple containing the responses.

    Raises:
        TypeError: If `to_send` is not a list of callables.
    """
    if not to_send or not all(isinstance(f, Callable) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    executor = Executor.get_instance()
    futures = [executor.submit(f, *args, **kwargs) for f in to_send]
    done, _ = concurrent.futures.wait(futures, timeout=timeout)

    responses: List[Any] = []
    for FUTURE in done:
        try:
            responses.append(FUTURE.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(None)
    return tuple(responses)


@trace("msgflow.nn.F.msg_bcast_gather")
def msg_bcast_gather(
    to_send: List[Callable],
    message: Message,
    response_mode: Optional[str] = "outputs",
    *,
    timeout: Optional[float] = None,
) -> Message:
    """
    Broadcasts a single message to multiple modules and gathers the responses.

    The given message is sent to each callable in `to_send`, and the responses are collected
    in the field specified by `response_mode`, using the module name as the key. Includes
    exception handling and optional timeout to prevent crashes.

    Args:
        to_send: List of callable objects (e.g. functions or `Module` instances).    
        message: Instance of `msgflow.Message` to broadcast.
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
    if not response_mode:
        raise ValueError("`response_mode` cannot be an empty string")

    executor = Executor.get_instance()
    futures = [executor.submit(f, message) for f in to_send]

    done, _ = concurrent.futures.wait(futures, timeout=timeout)
    for f, FUTURE in zip(to_send, done):
        f_name = get_callable_name(f)
        try:
            message.set(f"{response_mode}.{f_name}", FUTURE.result())
        except Exception as e:
            logger.error(f"Error in scattered task for `{f_name}`: {e}")
            message.set(f"{response_mode}.{f_name}", None)
    return message


@trace("msgflow.nn.F.wait_for")
def wait_for(to_send: Callable, *args, timeout: Optional[float] = None, **kwargs) -> Any:
    """
    Wait for a callable execution.

    Args:
        to_send: A callable object (e.g. functions or `Module` instances).    
        *args: Positional arguments.
        timeout: Maximum time (in seconds) to wait for responses.
        **kwargs: Named arguments.

    Returns:
        Callable responses.

    Raises:
        TypeError: If `to_send` is not a callable.
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    executor = Executor.get_instance()
    future = executor.submit(to_send, *args, **kwargs)
    done, _ = concurrent.futures.wait(future, timeout=timeout)    
    try:        
        return done[0].result()
    except Exception as e:
        logger.error(str(e))
        return None


@trace("msgflow.nn.F.wait_for_event")
def wait_for_event(event: asyncio.Event) -> None:
    """
    Waits synchronously for an asyncio.Event to be set.

    This function will block until event.set() is called elsewhere.

    Args:
        event: The asyncio.Event to wait for.

    Raises:
        TypeError: If `event` is not an instance of asyncio.Event.
    """
    if not isinstance(event, asyncio.Event):
        raise TypeError("`event` must be an instance of asyncio.Event")

    executor = Executor.get_instance()
    future = executor._submit_to_async_worker(event.wait())
    try:
        future.result()
    except Exception as e:
        logger.error(str(e))


@trace("msgflow.nn.F.background_task")
def background_task(
    to_send: Callable,
    *args,
    **kwargs
) -> None:
    """
    Executes a task in the background asynchronously without blocking, using the AsyncExecutorPool.
    This function is "fire-and-forget".

    Args:
        to_send: Callable object (function, async function, or module with .acall() method).
        *args: Positional arguments.
        **kwargs: Named arguments.

    Raises:
        TypeError: If `to_send` is not a callable.

    Examples:
        # Sync f
        background_task(my_function, arg1, arg2)
        
        # Async f
        async def async_func(x, y, z=None): ...
        background_task(async_func, 1, 2, z=3)
        
        # One param
        background_task(my_function, message)   
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    executor = Executor.get_instance()
    executor.submit(to_send, *args, **kwargs)
