# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Tuple
from msgflow.envs import envs
from msgflow.logger import logger
from msgflow.message import Message
from msgflow.nn.modules.module import get_callable_name
from msgflow.telemetry.span import trace


class AsyncExecutorPool:
    """
    Generic asynchronous executor pool that can be reused by
    different functions, avoiding excessive thread creation.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._thread_local = threading.local()
        self._lock = threading.Lock()
    
    def _get_executor(self):
        """Get thread-local executor to avoid deadlocks."""
        if not hasattr(self._thread_local, "executor"):
            with self._lock:
                if not hasattr(self._thread_local, "executor"):
                    self._thread_local.executor = ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix="AsyncMsgFlow"
                    )
        return self._thread_local.executor
    
    def run_async_function(self, async_func, *args, **kwargs):
        """
        Executes an asynchronous function using the thread pool.

        Args:
            async_func: Asynchronous function to execute
            *args, **kwargs: Arguments to the function

        Returns:
            Result of the asynchronous function
        """
        def run_in_thread():
            # Create new loop for this operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()
        
        try:            
            asyncio.get_running_loop()            
            executor = self._get_executor()
            future = executor.submit(run_in_thread)
            return future.result()
        except RuntimeError:
            # There is no loop running, use asyncio.run directly
            return asyncio.run(async_func(*args, **kwargs))
    
    def __del__(self):
        """Executor cleanup when object is destroyed."""
        if hasattr(self._thread_local, "executor"):
            self._thread_local.executor.shutdown(wait=False)

# Global
global _async_pool
_async_pool: Optional[AsyncExecutorPool] = None


def get_async_pool() -> AsyncExecutorPool:
    """Returns the async pool instance."""    
    if _async_pool is None:
        _async_pool = AsyncExecutorPool(max_workers=envs.num_threads_async_pool)
    return _async_pool


async def _execute_callable_async(callable_obj, message=None):
    """
    Executes a callable asynchronously.
    Checks if it has a .acall() method, otherwise it uses the callable directly.
    """
    try:
        if hasattr(callable_obj, "acall"):
            if message is not None:
                return await callable_obj.acall(message)
            else:
                return await callable_obj.acall()
        else:
            # If you don't have .acall(), run it in a thread pool to avoid blocking.
            loop = asyncio.get_event_loop()
            if message is not None:
                return await loop.run_in_executor(None, callable_obj, message)
            else:
                return await loop.run_in_executor(None, callable_obj)
    except Exception as e:
        logger.error(f"Error in execution of {get_callable_name(callable_obj)}: {e}")
        raise


def _run_background_task_fire_and_forget(
    message: Any,
    to_send: Callable,
    timeout: Optional[float] = None,
) -> None:
    """
    Executes a task in the background in a fire-and-forget manner using a daemon thread.
    It does not block and does not return results.
    """
    def run_background_thread():
        try:             
            loop = asyncio.new_event_loop() # Create new loop for this daemon thread
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_background_task_async(message, to_send, timeout))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Background thread failed: {e}")
    
    # Create daemon thread that does not block the main program
    thread = threading.Thread(target=run_background_thread, daemon=True)
    thread.start()


# Async back-ends


async def _background_task_async(
    message: Any,
    to_send: Callable,
    timeout: Optional[float] = None,
) -> None:
    """Async version of background_task."""
    if not isinstance(to_send, Callable):
        raise TypeError("`to_send` must be a callable object")
    
    try:
        if timeout is not None:
            await asyncio.wait_for(
                _execute_callable_async(to_send, message),
                timeout=timeout
            )
        else:
            await _execute_callable_async(to_send, message)
    except asyncio.TimeoutError:
        logger.error(f"Background task timeout after {timeout}s for {get_callable_name(to_send)}")
    except Exception as e:
        logger.error(f"Background task failed for {get_callable_name(to_send)}: {e}")


async def _bcast_gather_async(
    message: Any,
    to_send: List[Callable],
    timeout: Optional[float] = None,
) -> Tuple[Any]:
    """Async version of bcast_gather."""
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    tasks = [
        asyncio.create_task(_execute_callable_async(module, message))
        for module in to_send
    ]
    
    try:
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        for task in tasks: # Cancel pending tasks
            if not task.done():
                task.cancel()
        responses = [None] * len(tasks)

    processed_responses = []
    for response in responses:
        if isinstance(response, Exception):
            processed_responses.append(None)
        else:
            processed_responses.append(response)
    
    return tuple(processed_responses)


async def _scatter_gather_async(
    messages: List[Any],
    to_send: List[Callable],
    timeout: Optional[float] = None,
) -> Tuple[Any]:
    """Async version of scatter_gather."""
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")
    
    if len(messages) != len(to_send):
        raise ValueError(f"The size of `messages` ({len(messages)}) "
                        f"must be equal to that of `to_send`: ({len(to_send)})")

    tasks = [
        asyncio.create_task(_execute_callable_async(module, message))
        for module, message in zip(to_send, messages)
    ]
    
    try:
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        for task in tasks: # Cancel pending tasks
            if not task.done():
                task.cancel()
        responses = [None] * len(tasks)

    processed_responses = []
    for response in responses:
        if isinstance(response, Exception):
            processed_responses.append(None)
        else:
            processed_responses.append(response)
    
    return tuple(processed_responses)


async def _msg_bcast_gather_async(
    message: Message,
    to_send: List[Callable],
    response_mode: Optional[str] = "outputs",
    timeout: Optional[float] = None,
) -> Message:
    """Async version of msg_bcast_gather."""
    if not isinstance(message, Message):
        raise TypeError("`message` must be an instance of `msgflow.Message`")
    
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")
        
    if not isinstance(response_mode, str):
        raise TypeError(f"`response_mode` must be a string, but it was received `{type(response_mode)}`")
    if response_mode == "":
        raise ValueError("`response_mode` cannot be an empty string")
    
    tasks = [
        asyncio.create_task(_execute_callable_async(module, message))
        for module in to_send
    ]
    
    try:
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        for task in tasks: # Cancel pending tasks
            if not task.done():
                task.cancel()
        responses = [None] * len(tasks)
    
    for module, response in zip(to_send, responses):
        module_name = get_callable_name(module)
        if isinstance(response, Exception):
            message.set(f"{response_mode}.{module_name}", None)
        else:
            message.set(f"{response_mode}.{module_name}", response)
    
    return message


async def _msg_scatter_gather_async(
    messages: List[Message],
    to_send: List[Callable],
    response_mode: Optional[str] = "outputs",
    timeout: Optional[float] = None,
) -> Tuple[Message]:
    """Async version of msg_scatter_gather."""
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

    tasks = [
        asyncio.create_task(_execute_callable_async(module, msg))
        for module, msg in zip(to_send, messages)
    ]
    
    try:
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        for task in tasks: # Cancel pending tasks
            if not task.done():
                task.cancel()
        responses = [None] * len(tasks)
    
    for module, message, response in zip(to_send, messages, responses):
        module_name = get_callable_name(module)
        if isinstance(response, Exception):
            message.set(f"{response_mode}.{module_name}", None)
            logger.error(f"{module_name}: {response}")
        else:
            message.set(f"{response_mode}.{module_name}", response)
    
    return tuple(messages)


# Sync front-ends

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
    async_pool = get_async_pool()
    return async_pool.run_async_function(_bcast_gather_async, message, to_send, timeout)


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
    async_pool = get_async_pool()
    return async_pool.run_async_function(_msg_bcast_gather_async, message, to_send, response_mode, timeout)


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
    async_pool = get_async_pool()
    return async_pool.run_async_function(_scatter_gather_async, messages, to_send, timeout)


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
    async_pool = get_async_pool()    
    return async_pool.run_async_function(_msg_scatter_gather_async, messages, to_send, response_mode, timeout)


@trace("msgflow.nn.F.background_task")
def background_task(
    message: Any,
    to_send: Callable,
    timeout: Optional[float] = None,
) -> None:
    """
    Executes a task in the background asynchronously without blocking.

    This function is "fire-and-forget" - it starts execution and returns immediately,
    without waiting for the result. If an error occurs, it just logs and continues.

    Args:
        message: Data to send to the callable. If None, call without parameters.
        to_send: Callable object (function or module with .acall() method).
        timeout: Maximum time (in seconds) to wait for execution.

    Raises:
        TypeError: If `to_send` is not a callable.
    """
    if not isinstance(to_send, Callable):
        raise TypeError("`to_send` must be a callable object")
    
    # Fire-and-forget execution
    _run_background_task_fire_and_forget(message, to_send, timeout)
