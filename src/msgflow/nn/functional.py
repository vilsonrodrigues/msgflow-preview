# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple
from msgflow._private.pool import get_async_pool
from msgflow.logger import logger
from msgflow.message import Message
from msgflow.nn.modules.module import get_callable_name
from msgflow.telemetry.span import trace



async def _execute_callable_async(callable_obj, *args, **kwargs):
    """
    Executes a callable asynchronously.
    Checks if it has a .acall() method, is an async function, otherwise runs in thread pool.
    """
    try:
        if hasattr(callable_obj, "acall"):
            return await callable_obj.acall(*args, **kwargs) 
        elif inspect.iscoroutinefunction(callable_obj): # async def
            return await callable_obj(*args, **kwargs)
        else: # sync def, run in thread pool (default executor of the current loop)
            loop = asyncio.get_event_loop()
            # For synchronous functions, loop.run_in_executor executes them in a thread pool
            # (the loop's default or one specified), allowing the loop to continue.            
            return await loop.run_in_executor(None, callable_obj, *args, **kwargs)
            
    except Exception as e:
        logger.error(f"Error in execution of {get_callable_name(callable_obj)}: {e}")
        raise


# Async back-ends


async def _wait_for_event_async(event: asyncio.Event):
    """Asynchronous helper to wait for an asyncio.Event."""
    await event.wait()


async def _background_task_async(
    to_send: Callable,
    *args,
    timeout: Optional[float] = None,
    **kwargs
) -> None:
    """Async version of background_task with support for args and kwargs."""
    try:
        if timeout is not None:
            await asyncio.wait_for(
                _execute_callable_async(to_send, *args, **kwargs),
                timeout=timeout
            )
        else:
            await _execute_callable_async(to_send, *args, **kwargs)
    except asyncio.TimeoutError:
        logger.error(f"Background task timeout after {timeout}s for {get_callable_name(to_send)}")
    except Exception as e:        
        logger.error(f"Background task failed for {get_callable_name(to_send)}: {e}")


async def _bcast_gather_async(
    to_send: List[Callable],
    *args,
    timeout: Optional[float] = None,
    **kwargs
) -> Tuple[Any, ...]:
    """Async version of bcast_gather."""
    tasks = [
        asyncio.create_task(_execute_callable_async(module, *args, **kwargs))
        for module in to_send
    ]

    responses: List[Any]
    try:
        if timeout is not None:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.TimeoutError:
        logger.warning(f"Broadcast gather timed out after {timeout}s. Cancelling tasks.")
        for task in tasks: # Cancel pending tasks
            if not task.done():
                task.cancel()
        # Wait for tasks to process the cancellation
        # CancelledError exceptions will be caught in responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    processed_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            if isinstance(response, asyncio.CancelledError):
                logger.warning(f"Task {i} for {get_callable_name(to_send[i])} was cancelled due to timeout.")
            else:
                logger.error(f"Error in gathered task {i} for {get_callable_name(to_send[i])}: {response}")
            processed_responses.append(None)
        else:
            processed_responses.append(response)

    return tuple(processed_responses)


async def _scatter_gather_async(
    to_send: List[Callable],
    args_list: Optional[List[Tuple[Any, ...]]] = None,
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    *,
    timeout: Optional[float] = None,
) -> Tuple[Any, ...]:
    """Async version of scatter_gather."""
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` deve ser uma lista não vazia de objetos chamáveis.")

    if args_list is not None and len(to_send) != len(args_list):
        raise ValueError(
            f"O comprimento de `to_send` ({len(to_send)}) deve corresponder ao comprimento de `args_list` ({len(args_list)})."
        )
    if kwargs_list is not None and len(to_send) != len(kwargs_list):
        raise ValueError(
            f"O comprimento de `to_send` ({len(to_send)}) deve corresponder ao comprimento de `kwargs_list` ({len(kwargs_list)})."
        )
    
    tasks = []
    for i, module in enumerate(to_send):
        current_args: Tuple[Any, ...] = args_list[i] if args_list and i < len(args_list) else ()
        current_kwargs: Dict[str, Any] = kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {}
        tasks.append(asyncio.create_task(_execute_callable_async(module, *current_args, **current_kwargs)))

    responses: List[Any]
    try:
        if timeout is not None:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.TimeoutError:
        logger.warning(f"scatter_gather had timeout after {timeout}s. Canceling tasks.")
        for task in tasks:
            if not task.done():
                task.cancel()
        responses = await asyncio.gather(*tasks, return_exceptions=True) # Allows cancellations to be processed

    processed_responses = []
    for i, response in enumerate(responses):
        callable_name = get_callable_name(to_send[i])
        
        log_args_info = ""
        if args_list and i < len(args_list):
            log_args_info += f"args={str(args_list[i])[:50]}" # Limita o tamanho para o log
        if kwargs_list and i < len(kwargs_list):
            if log_args_info: log_args_info += ", "
            log_args_info += f"kwargs={str(kwargs_list[i])[:50]}"

        if isinstance(response, Exception):
            if isinstance(response, asyncio.CancelledError):
                logger.warning(f"Task to {callable_name} with ({log_args_info}) was canceled due to timeout.")
            else:
                logger.error(f"Error in task spread to {callable_name} with ({log_args_info}): {response}")
            processed_responses.append(None)
        else:
            processed_responses.append(response)

    return tuple(processed_responses)


async def _msg_bcast_gather_async(
    to_send: List[Callable],
    message: Message,
    response_mode: Optional[str] = "outputs",
    timeout: Optional[float] = None,
) -> Message:
    """Async version of msg_bcast_gather."""
    tasks = [
        asyncio.create_task(_execute_callable_async(module, message))
        for module in to_send
    ]

    responses: List[Any]
    try:
        if timeout is not None:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.TimeoutError:
        logger.warning(f"Message broadcast gather timed out after {timeout}s. Cancelling tasks.")
        for task in tasks: # Cancel pending tasks
            if not task.done():
                task.cancel()
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for module, response in zip(to_send, responses):
        module_name = get_callable_name(module)
        if isinstance(response, Exception):
            if isinstance(response, asyncio.CancelledError):
                 logger.warning(f"Task for {module_name} (msg_bcast) was cancelled.")
            else:
                logger.error(f"Error in gathered task for {module_name} (msg_bcast): {response}")
            message.set(f"{response_mode}.{module_name}", None)
        else:
            message.set(f"{response_mode}.{module_name}", response)

    return message


async def _msg_scatter_gather_async(
    to_send: List[Callable],
    messages: List[Message],
    response_mode: Optional[str] = "outputs",
    timeout: Optional[float] = None,
) -> Tuple[Message, ...]:
    """Async version of msg_scatter_gather."""
    tasks = [
        asyncio.create_task(_execute_callable_async(module, msg))
        for module, msg in zip(to_send, messages)
    ]

    responses: List[Any]
    try:
        if timeout is not None:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            responses = await asyncio.gather(*tasks, return_exceptions=True) # CORRIGIDO: Adicionado await
    except asyncio.TimeoutError:
        logger.warning(f"Message scatter gather timed out after {timeout}s. Cancelling tasks.")
        for task in tasks: # Cancel pending tasks
            if not task.done():
                task.cancel()
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for module, message_item, response in zip(to_send, messages, responses):
        module_name = get_callable_name(module)
        if isinstance(response, Exception):
            if isinstance(response, asyncio.CancelledError):
                logger.warning(f"Task for {module_name} with msg id {message_item.id if hasattr(message_item, 'id') else 'N/A'} (msg_scatter) was cancelled.")
            else:
                logger.error(f"Error in scattered task for {module_name} with msg id {message_item.id if hasattr(message_item, 'id') else 'N/A'} (msg_scatter): {response}")

            message_item.set(f"{response_mode}.{module_name}", None)
        else:
            message_item.set(f"{response_mode}.{module_name}", response)

    return tuple(messages)

# Sync front-ends

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

    async_pool = get_async_pool()
    return async_pool.run_async_function(_bcast_gather_async, to_send, *args, timeout=timeout, **kwargs)


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
    
    async_pool = get_async_pool()
    return async_pool.run_async_function(_msg_bcast_gather_async, to_send, message, response_mode, timeout)


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
        to_send: List of callable objects (e.g. functions or `Module` instances).
        args_list: Optional list of tuples. Each tuple contains the positional arguments
            for the corresponding callable in `to_send`. If `None`, no positional arguments 
            are passed unless specified individually by an item in `kwargs_list`.
        kwargs_list: Optional list of dictionaries. Each dictionary contains the named arguments 
            for the corresponding callable in `to_send`. If `None`, no named arguments are passed 
            unless specified individually by an item in `args_list`.
        timeout: Maximum time (in seconds) to wait for responses.

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

        # Example 4: Calling functions that take no arguments (or using all defaults)
        import random
        def get_random_num(): return random.randint(0,100)
        def get_another_random(): return random.random()
        random_funcs = [get_random_num, get_another_random, get_random_num]
        results = scatter_gather(random_funcs) # args_list and kwargs_list are None
        print(len(results))
        # To be explicit about not passing args/kwargs to each:
        explicit_args = [(), (), ()]
        explicit_kwargs = [{}, {}, {}]
        results_explicit = scatter_gather(random_funcs, args_list=explicit_args, kwargs_list=explicit_kwargs)
        print(len(results_explicit)) # Saída esperada: 3
    """
    if not isinstance(to_send, list) or not all(callable(f) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    async_pool = get_async_pool()
    return async_pool.run_async_function(
        _scatter_gather_async,
        to_send,
        args_list=args_list,
        kwargs_list=kwargs_list,
        timeout=timeout
    )


@trace("msgflow.nn.F.msg_scatter_gather")
def msg_scatter_gather(
    to_send: List[Callable],
    messages: List[Message],
    *,
    response_mode: Optional[str] = "outputs",    
    timeout: Optional[float] = None,
) -> Tuple[Message]:
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

    async_pool = get_async_pool()    
    return async_pool.run_async_function(_msg_scatter_gather_async, to_send, messages, response_mode, timeout)


@trace("msgflow.nn.F.background_task")
def background_task(
    to_send: Callable,
    *args,    
    timeout: Optional[float] = None,
    **kwargs
) -> None:
    """
    Executes a task in the background asynchronously without blocking, using the AsyncExecutorPool.
    This function is "fire-and-forget".

    Args:
        to_send: Callable object (function, async function, or module with .acall() method).
        *args: Positional arguments.
        timeout: Maximum time (in seconds) to wait for responses.
        **kwargs: Named arguments.

    Raises:
        TypeError: If `to_send` is not a callable.

    Examples:
        # Sync fn
        background_task(my_function, arg1, arg2, timeout=10.0)
        
        # Async fn
        async def async_func(x, y, z=None): ...
        background_task(async_func, 1, 2, z=3)
        
        # One param
        background_task(my_function, message)    
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    async_pool = get_async_pool()

    def run_pooled_background_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_background_task_async(to_send, *args, timeout=timeout, **kwargs))
        except Exception as e:
            logger.error(
                f"Outer exception in pooled background task for {get_callable_name(to_send)}: {e}"
            )
        finally:
            loop.close()

    async_pool.submit_to_pool(run_pooled_background_task)


@trace("msgflow.nn.F.wait_for")
def wait_for(event: asyncio.Event) -> None:
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
        
    async_pool = get_async_pool()
    async_pool.run_async_function(_wait_for_event_async, event)
