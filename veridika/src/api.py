import os
from typing import Any


def get_api_key(API_KEY_NAME: str) -> str:
    """
    Returns the API key
    """

    if os.path.exists(f"/run/secrets/{API_KEY_NAME}"):
        with open(f"/run/secrets/{API_KEY_NAME}") as f:
            return f.read().strip()

    api_key = os.getenv(API_KEY_NAME)
    if api_key is not None:
        return api_key.strip()

    raise KeyError(
        f"API key {API_KEY_NAME} not found. Please set it as an environment variable or a docker secret."
    )


class ApiHandler:
    def __init__(self, api_name: str):
        self.api_name = api_name
        self._cost = []

    @property
    def cost(self):
        return self.get_cost()

    def get_cost(self) -> float:
        """
        Returns the accumulated cost of the API calls
        """
        return sum(self._cost)

    def reset_cost(self):
        """
        Resets the accumulated cost of the API calls
        """
        self._cost = []

    def get_last_cost(self) -> float:
        """
        Returns the cost of the last API call
        """
        if len(self._cost) == 0:
            raise ValueError("No API calls have been made yet")
        return self.cost[-1]

    def add_cost(self, cost: float):
        """
        Add the cost of the last API call to the accumulated cost
        """
        self._cost.append(cost)

    def __call__(self, *args, **kwargs) -> tuple[Any, float]:
        """
        Call the API

        Returns:
            Any: The result of the API call
            float: The cost of the API call
        """
        raise NotImplementedError("This method should be implemented by the child class")
