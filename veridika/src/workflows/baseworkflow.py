from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any

from omegaconf import OmegaConf

from veridika.src.agents.baseagent import BaseAgent
from veridika.src.managers.cication_manager import CitationManager


class BaseWorkflow(ABC):
    """Base class for all fact-checking workflows.

    This class provides common functionality for managing agents, tracking costs,
    loading configuration, and handling async execution patterns.
    """

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        """Initialize the base workflow.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
            config_path (Optional[str]): Path to YAML configuration file
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config or config_path must be provided")

        # Initialize tracking variables
        self._agents: list[BaseAgent] = []
        self._start_time: float = 0.0
        self._total_runtime: float = 0.0
        self._citation_manager: CitationManager | None = None

        # Initialize agents (implemented by subclasses)
        self._initialize_agents()

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file using OmegaConf.

        Args:
            config_path (str): Path to YAML configuration file

        Returns:
            Dict[str, Any]: Loaded configuration
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load with OmegaConf for advanced features like nested configs and interpolation
        omega_config = OmegaConf.load(config_file)
        # Convert to regular dict to maintain compatibility with existing code
        return OmegaConf.to_container(omega_config, resolve=True)

    @abstractmethod
    def _initialize_agents(self) -> None:
        """Initialize all agents for this workflow (implemented by subclasses)."""
        pass

    @abstractmethod
    async def run(self, statement: str, **kwargs) -> tuple[dict[str, Any], float]:
        """Execute the workflow (implemented by subclasses).

        Args:
            statement (str): The fact-checking statement or question
            **kwargs: Additional workflow-specific parameters

        Returns:
            Dict[str, Any]: Formatted workflow output
            float: Total cost of the workflow
        """
        pass

    def _start_timing(self) -> None:
        """Start timing the workflow execution."""
        self._start_time = time.perf_counter()

    def _stop_timing(self) -> None:
        """Stop timing and record total runtime."""
        self._total_runtime = time.perf_counter() - self._start_time

    def get_citation_manager(self, new_manager: bool = True) -> CitationManager:
        """Get or create a shared citation manager.

        Returns:
            CitationManager: Shared citation manager instance
        """
        if self._citation_manager is None or new_manager:
            self._citation_manager = CitationManager()
        return self._citation_manager

    def get_total_cost(self) -> float:
        """Calculate total cost across all agents.

        Returns:
            float: Total cost of all agent operations
        """
        return sum(agent.cost for agent in self._agents)

    def get_total_calls(self) -> int:
        """Calculate total calls across all agents.

        Returns:
            int: Total number of agent calls
        """
        return sum(agent.calls for agent in self._agents)

    def get_runtime(self) -> float:
        """Get total workflow runtime.

        Returns:
            float: Total runtime in seconds
        """
        return self._total_runtime

    def get_agents(self) -> list[BaseAgent]:
        """Get list of all agents in this workflow.

        Returns:
            List[BaseAgent]: List of all agents
        """
        return self._agents.copy()

    def get_workflow_stats(self) -> dict[str, Any]:
        """Get comprehensive workflow statistics.

        Returns:
            Dict[str, Any]: Workflow statistics including costs, calls, and runtime
        """
        return {
            "total_cost": self.get_total_cost(),
            "total_calls": self.get_total_calls(),
            "total_runtime": self._total_runtime,
            "agents_count": len(self._agents),
            "agent_stats": {agent.name: agent.get_stats() for agent in self._agents},
        }

    def reset_all_agents(self) -> None:
        """Reset all agents to their initial state."""
        for agent in self._agents:
            agent.reset()
        self._citation_manager = None
        self._total_runtime = 0.0

    def _add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the workflow's agent list.

        Args:
            agent (BaseAgent): Agent to add
        """
        self._agents.append(agent)


class BaseStepwiseWorkflow(ABC):
    """Base class for simple workflows that don't require agents to be registered or a configuration file.

    This class doesn't register the agents in the workflow, it only tracks the total cost, runtime and calls.
    This is useful for stepwise workflows in which we might use a different model each run/step.
    """

    def __init__(self, step_name: str):
        """Initialize the base workflow.

        Args:
            step_name (str): The name of the step
        """
        # Load configuration
        self.step_name = step_name
        self.total_cost = 0.0
        self.total_runtime = 0.0
        self.total_calls = 0

    @abstractmethod
    async def run(
        self, statement: str, language: str, location: str, **kwargs
    ) -> tuple[dict[str, Any], float]:
        """Execute the workflow (implemented by subclasses).

        Args:
            statement (str): The fact-checking statement or question
            language (str): The language of the statement
            location (str): The location of the statement
            **kwargs: Additional workflow-specific parameters

        Returns:
            Dict[str, Any]: Formatted workflow output
            float: Total cost of the workflow
        """
        pass

    def _start_timing(self) -> None:
        """Start timing the workflow execution."""
        self._start_time = time.perf_counter()

    def _stop_timing(self) -> None:
        """Stop timing and record total runtime."""
        self._total_runtime = time.perf_counter() - self._start_time

    def get_total_cost(self) -> float:
        """Calculate total cost across all agents.

        Returns:
            float: Total cost of all agent operations
        """
        return self.total_cost

    def get_total_calls(self) -> int:
        """Calculate total calls across all agents.

        Returns:
            int: Total number of agent calls
        """
        return self.total_calls

    def get_runtime(self) -> float:
        """Get total workflow runtime.

        Returns:
            float: Total runtime in seconds
        """
        return self.total_runtime

    def get_workflow_stats(self) -> dict[str, Any]:
        """Get comprehensive workflow statistics.

        Returns:
            Dict[str, Any]: Workflow statistics including costs, calls, and runtime
        """
        return {
            "total_cost": self.get_total_cost(),
            "total_calls": self.get_total_calls(),
            "total_runtime": self._total_runtime,
        }
