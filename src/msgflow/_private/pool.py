import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine, Optional
from msgflow.envs import envs


class AsyncExecutorPool:
    """
    Generic asynchronous executor pool that can be reused by
    different functions, avoiding excessive thread creation.
    This pool manages a single, shared ThreadPoolExecutor.
    """

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="AsyncMsgFlowPool"
        )

    def run_async_function(self, async_func: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs) -> Any:
        """
        Executes an asynchronous function using the shared thread pool.
        This function will block the caller until the async_func completes.

        Args:
            async_func: Asynchronous function to execute.
            *args, **kwargs: Arguments to the function.

        Returns:
            Result of the asynchronous function.
        """
        def run_in_thread(): # Each async function runs in its own event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()

        future = self._executor.submit(run_in_thread)
        return future.result()

    def submit_to_pool(self, func_to_run_in_thread: Callable[[], Any]) -> None:
        """
        Submits a synchronous function (which may internally manage an async loop)
        to the pool for fire-and-forget execution.
        `func_to_run_in_thread` is expected to handle its own exceptions.
        """
        self._executor.submit(func_to_run_in_thread)

    def __del__(self):
        """Executor cleanup when object is destroyed."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=False)

# Global
_async_pool_lock = threading.Lock()
_async_pool: Optional[AsyncExecutorPool] = None


def configure_async_pool():
    """Configures async thread pool."""
    if _async_pool is None:
        with _async_pool_lock:
            _async_pool = AsyncExecutorPool(max_workers=envs.num_threads_async_pool)    


def get_async_pool() -> AsyncExecutorPool:
    """Returns the async pool instance (singleton)."""
    configure_async_pool()
    return _async_pool
