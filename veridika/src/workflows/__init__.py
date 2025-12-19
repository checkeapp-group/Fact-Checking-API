"""Fact-checking workflows package.

This package contains workflow classes that orchestrate multiple agents
to perform comprehensive fact-checking analysis.
"""

from typing import Any

from .baseworkflow import BaseWorkflow

__all__ = [
    "BaseWorkflow",
    "FactCheckingWithPipelineWorkflow",
]


class Workflow(BaseWorkflow):
    """
    Factory-style wrapper that yields either a `FactCheckingWithPipelineWorkflow`
    instance depending on *workflow_type*.

    Example
    -------
    ```python
    workflow = Workflow(
        "FactCheckingWithPipelineWorkflow"
    )  # -> actually a FactCheckingWithPipelineWorkflow instance
    other = Workflow("other_workflow")  # -> raises ValueError
    ```
    """

    def __new__(cls, config: dict[str, Any] | None = None, config_path: str | None = None):
        if config_path:
            config = BaseWorkflow._load_config(config_path)
        elif config:
            config = config
        else:
            raise ValueError("Either config or config_path must be provided")

        workflow_type = config.get("workflow_type", "Unknown")
        if workflow_type == "FactCheckingWithPipelineWorkflow":
            from .FactCheckingWorkflow import (
                FactCheckingWithPipelineWorkflow,
            )

            return FactCheckingWithPipelineWorkflow(config=config)
        else:
            if workflow_type == "Unknown":
                raise ValueError(
                    "Workflow type is not specified in the config file. Please specify a workflow type in the config file using the 'workflow_type' key."
                )
            else:
                raise ValueError(f"Invalid workflow type: {workflow_type}")
