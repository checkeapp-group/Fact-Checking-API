from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.image import Image


class ImageGenAgent(BaseAgent):
    def __init__(self, max_history_size: int | None = None, image_model: str = "flux_replicate"):
        """Initialize the ImageGenAgent.

        Args:
            max_history_size (Optional[int]): Maximum size of conversation history to maintain
            image_model (str): Image model identifier passed to the Image factory

        Returns:
            None
        """
        super().__init__(
            name="ImageGenAgent",
            description="An agent that generates images from descriptions using Flux image generator",
            max_history_size=max_history_size,
        )
        self.image_model = image_model
        self.image_api = Image(self.image_model)

    def run(
        self,
        image_description_output: dict[str, str],
        size: str = "1280x256",
        style: str = "The style is bold and energetic, using vibrant colors and dynamic compositions to create visually engaging scenes that emphasize activity, culture, and lively atmospheres. Digital art.",
    ) -> tuple[dict[str, str], float, HistoryEntry]:
        """Generate an image from the provided image description.

        Args:
            image_description_output (Dict[str, str]): Output from ImagePromptAgent containing {"image_description": "description text"}
            size (str): Image size in "WIDTHxHEIGHT" format (default: "1280x256")
            **kwargs: Additional parameters to pass to Flux image generator

        Returns:
            Tuple[Dict[str, str], float, HistoryEntry]: A tuple containing:
                - Dictionary with image URL: {"image_url": "generated_image_url"}
                - The cost of the image generation operation
                - History entry containing operation metadata
        """
        # Extract image description from input
        image_description = image_description_output.get("image_description", "")

        if not image_description:
            raise ValueError("No image description provided in input")

        image_description = f"{image_description} {style}"

        # Generate image using configured image API (Replicate Flux or ConfiUI)
        image_result, cost = self.image_api(
            image_description=image_description,
            size=size,
        )

        # Normalise output: if base64, convert to a data URL for compatibility
        if isinstance(image_result, str) and not image_result.lower().startswith("http"):
            # Assume PNG for ComfyUI output
            image_url = f"data:image/png;base64,{image_result}"
        else:
            image_url = image_result

        # Create result dictionary
        result = {"image_url": image_url}

        # Create history entry
        history_entry = HistoryEntry(
            data={
                "image_description": image_description,
                "size": size,
                "image_url": image_url,
            },
            type=HistoryEntryType.image_generation,
            model_metadata={
                "model_name": getattr(self.image_api, "api_name", self.image_model),
                "agent_type": "image_generator",
                "image_description": image_description,
                "size": size,
                "generation_cost": cost,
            },
        )

        return result, cost, history_entry
