import asyncio
from collections import deque
from datetime import datetime
from enum import Enum
import json
from threading import Lock
import time
from typing import Any


class HistoryEntryType(Enum):
    """Enumeration of possible history entry types for agent operations."""

    conversation = "conversation"
    image_generation = "image_generation"
    rag = "rag"
    web_search = "web_search"


class HistoryEntry:
    """Represents a single entry in an agent's execution history.

    This class encapsulates the data and type information for operations
    performed by an agent, providing serialization capabilities.
    """

    def __init__(
        self,
        data: str | dict[str, Any],
        type: HistoryEntryType,
        model_metadata: dict[str, Any] | None = None,
    ):
        """Initialize a new history entry.

        Args:
            data (str): The data or content associated with this history entry.
            type (HistoryEntryType): The type/category of this history entry.
        """
        self.data = data
        self.type = type
        self.model_metadata = model_metadata

    def __str__(self) -> str:
        """Convert the history entry to a JSON string representation.

        Returns:
            str: JSON string representation of the history entry.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the history entry to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing 'data' and 'type' keys.
        """
        return {
            "data": self.data,
            "type": self.type.value,
        }


class BaseAgent:
    """Base class for asynchronous AI agents with history tracking and statistics.

    This class provides a foundation for implementing AI agents that can run
    asynchronously, track execution history, maintain statistics, and handle
    concurrent operations safely.
    """

    def __init__(self, name: str, description: str, max_history_size: int | None = None):
        """Initialize a new BaseAgent instance.

        Args:
            name (str): Unique name identifier for this agent.
            description (str): Human-readable description of the agent's purpose.
            max_history_size (Optional[int], optional): Maximum number of history
                entries to retain. If None, history will grow unbounded.
                Defaults to None.
        """
        self.name = name
        self.description = description
        self.max_history_size = max_history_size

        # Thread-safe counters and state
        self._lock = Lock()
        self._total_cost = 0.0
        self._total_calls = 0
        self._run_time = 0.0
        self._error_count = 0

        # Use deque for efficient history management
        self._history = deque(maxlen=max_history_size)

    @property
    def cost(self) -> float:
        """Get the total accumulated cost across all agent calls.

        Returns:
            float: Total cost incurred by this agent across all operations.
        """
        with self._lock:
            return self._total_cost

    @property
    def calls(self) -> int:
        """Get the total number of calls made to this agent.

        Returns:
            int: Total number of times this agent has been called.
        """
        with self._lock:
            return self._total_calls

    @property
    def total_run_time(self) -> float:
        """Get the total accumulated runtime across all agent calls.

        Returns:
            float: Total runtime in seconds for all agent operations.
        """
        with self._lock:
            return self._run_time

    def run(self, *args, **kwargs) -> tuple[Any, float, HistoryEntry]:
        """Execute the agent's core functionality (must be implemented by subclasses).

        This is the main method that subclasses must implement to define their
        specific behavior. It will be called in a thread pool for async execution.

        Args:
            *args: Variable length argument list passed to the agent.
            **kwargs: Arbitrary keyword arguments passed to the agent.

        Returns:
            Tuple[Any, float, HistoryEntry]: A tuple containing:
                - result: The output/result of the agent's operation
                - cost: The monetary or computational cost of this operation
                - history_entry: A HistoryEntry object documenting this operation

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _add_to_history(self, history_entry: HistoryEntry, run_time: float, cost: float):
        """Add an entry to the agent's execution history in a thread-safe manner.

        Args:
            history_entry (HistoryEntry): The history entry to add.
            run_time (float): The execution time for this operation in seconds.
            cost (float): The cost associated with this operation.
        """
        history_record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_time": run_time,
            "cost": cost,
            "history_entry": history_entry.to_dict(),
        }

        with self._lock:
            self._history.append(history_record)

    def get_history(self) -> list[dict[str, Any]]:
        """Get a thread-safe snapshot of the agent's execution history.

        Returns:
            List[Dict[str, Any]]: A list of history records, where each record
                contains 'date', 'run_time', 'cost', and 'history_entry' keys.
        """
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        """Reset the agent's statistics and history. Factory reset of the agent.

        This operation is thread-safe and will reset all accumulated statistics
        and clear the execution history.
        """
        with self._lock:
            self._history.clear()
            self._total_cost = 0.0
            self._total_calls = 0
            self._run_time = 0.0
            self._error_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the agent's performance.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - name: Agent's name
                - total_calls: Total number of calls made
                - total_cost: Total accumulated cost
                - total_run_time: Total accumulated runtime in seconds
                - average_cost_per_call: Average cost per operation
                - average_time_per_call: Average time per operation in seconds
                - history_size: Current number of entries in history
        """
        with self._lock:
            return {
                "name": self.name,
                "error_count": self._error_count,
                "total_calls": self._total_calls,
                "total_cost": self._total_cost,
                "total_run_time": self._run_time,
                "average_cost_per_call": self._total_cost / max(self._total_calls, 1),
                "average_time_per_call": self._run_time / max(self._total_calls, 1),
                "history_size": len(self._history),
            }

    async def _call(self, *args, **kwargs) -> tuple[Any, float]:
        """Internal async wrapper for the synchronous run method with error handling.

        This method handles the async execution of the synchronous run method,
        including timing, error handling, statistics tracking, and history management.

        Args:
            *args: Variable length argument list to pass to the run method.
            **kwargs: Arbitrary keyword arguments to pass to the run method.

        Returns:
            Tuple[Any, float]: A tuple containing:
                - result: The output from the run method
                - cost: The cost associated with this operation

        Raises:
            Exception: Re-raises any exception that occurs during execution,
                but ensures statistics and history are still updated.
        """
        # More accurate timing
        start_time = time.perf_counter()

        try:
            # Run the synchronous method in a thread pool
            result, cost, history_entry = await asyncio.to_thread(self.run, *args, **kwargs)

            # Calculate timing
            current_run_time = time.perf_counter() - start_time

            # Thread-safe state updates
            with self._lock:
                self._total_calls += 1
                self._total_cost += cost
                self._run_time += current_run_time

            # Add to history
            self._add_to_history(history_entry=history_entry, run_time=current_run_time, cost=cost)

            return result, cost

        except Exception:
            # Still update call count on error for accurate stats
            current_run_time = time.perf_counter() - start_time
            with self._lock:
                self._total_calls += 1
                self._run_time += current_run_time
                self._error_count += 1

            # Re-raise the exception
            raise

    def __call__(self, *args, **kwargs) -> asyncio.Task[tuple[Any, float]]:
        """Make the agent callable, returning a task for deferred execution.

        This method allows the agent to be called like a function while returning
        an asyncio.Task that can be awaited later. This enables flexible async
        orchestration where multiple agents can run concurrently and their
        results can be awaited when needed.

        Args:
            *args: Variable length argument list to pass to the agent's run method.
            **kwargs: Arbitrary keyword arguments to pass to the agent's run method.

        Returns:
            asyncio.Task[Tuple[Any, float]]: An asyncio Task that when awaited
                returns a tuple of (result, cost).

        Example:
            >>> agent = MyAgent("test", "description")
            >>> task1 = agent("input1")  # Returns immediately, starts running
            >>> task2 = agent("input2")  # Returns immediately, starts running
            >>> result1, cost1 = await task1  # Wait for first result
            >>> result2, cost2 = await task2  # Wait for second result
        """
        return asyncio.create_task(self._call(*args, **kwargs))
