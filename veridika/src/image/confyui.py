from __future__ import annotations

import json
import time
from typing import Any

import requests

from veridika.src.api import ApiHandler, get_api_key

_SIZE_RE = __import__("re").compile(r"^\s*(\d+)[xX](\d+)\s*$")


class ConfiUI(ApiHandler):
    """
    Runpod ComfyUI image generator wrapper.

    Returns a base64-encoded image and a cost (0.0 for local/hosted endpoints).
    """

    def __init__(
        self,
        model: str,
        runpod_url_env_name: str,
        api_key_env_name: str,
        pretty_name: str | None = None,
    ) -> None:
        self.model = model
        try:
            self.run_url = get_api_key(runpod_url_env_name)
        except KeyError:
            raise ValueError(f"Runpod URL environment variable {runpod_url_env_name} not found")

        try:
            self.api_key = get_api_key(api_key_env_name)
        except KeyError:
            raise ValueError(f"Runpod API key environment variable {api_key_env_name} not found")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        super().__init__(model if pretty_name is None else pretty_name)

    # ----------------------------------------------------------
    # utility helpers
    # ----------------------------------------------------------
    @staticmethod
    def _parse_size(size: str) -> tuple[int, int]:
        m = _SIZE_RE.match(size)
        if not m:
            raise ValueError(
                f"Size must be of the form 'WIDTHxHEIGHT', e.g. '1024x1024' (got {size!r})"
            )
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def _build_workflow(prompt: str, width: int, height: int) -> dict[str, Any]:
        """
        Build the ComfyUI workflow payload, customizing prompt and size.
        """
        template: dict[str, Any] = {
            "input": {
                "workflow": {
                    "6": {
                        "inputs": {
                            "text": prompt,
                            "clip": ["30", 1],
                        },
                        "class_type": "CLIPTextEncode",
                    },
                    "8": {
                        "inputs": {
                            "samples": ["31", 0],
                            "vae": ["30", 2],
                        },
                        "class_type": "VAEDecode",
                    },
                    "9": {
                        "inputs": {
                            "filename_prefix": "ComfyUI",
                            "images": ["8", 0],
                        },
                        "class_type": "SaveImage",
                    },
                    "27": {
                        "inputs": {
                            "width": width,
                            "height": height,
                            "batch_size": 1,
                        },
                        "class_type": "EmptySD3LatentImage",
                    },
                    "30": {
                        "inputs": {
                            "ckpt_name": "flux1-dev-fp8.safetensors",
                        },
                        "class_type": "CheckpointLoaderSimple",
                    },
                    "31": {
                        "inputs": {
                            "seed": 972054013131368,
                            "steps": 20,
                            "cfg": 1.0,
                            "sampler_name": "euler",
                            "scheduler": "simple",
                            "denoise": 1.0,
                            "model": ["30", 0],
                            "positive": ["35", 0],
                            "negative": ["33", 0],
                            "latent_image": ["27", 0],
                        },
                        "class_type": "KSampler",
                    },
                    "33": {
                        "inputs": {
                            "text": "",
                            "clip": ["30", 1],
                        },
                        "class_type": "CLIPTextEncode",
                    },
                    "35": {
                        "inputs": {
                            "guidance": 3.5,
                            "conditioning": ["6", 0],
                        },
                        "class_type": "FluxGuidance",
                    },
                },
                "workflow_meta": {
                    "last_node_id": 36,
                    "last_link_id": 57,
                    "version": 0.4,
                    # Note: UI-only metadata retained verbatim for compatibility
                    "nodes": [
                        {
                            "id": 33,
                            "type": "CLIPTextEncode",
                            "pos": [390, 400],
                            "size": [422.84503173828125, 164.31304931640625],
                            "flags": {"collapsed": True},
                            "order": 4,
                            "mode": 0,
                            "inputs": [
                                {"name": "clip", "type": "CLIP", "link": 54, "slot_index": 0}
                            ],
                            "outputs": [
                                {
                                    "name": "CONDITIONING",
                                    "type": "CONDITIONING",
                                    "links": [55],
                                    "slot_index": 0,
                                }
                            ],
                            "title": "CLIP Text Encode (Negative Prompt)",
                            "properties": {"Node name for S&R": "CLIPTextEncode"},
                            "widgets_values": [""],
                            "color": "#322",
                            "bgcolor": "#533",
                        },
                        {
                            "id": 27,
                            "type": "EmptySD3LatentImage",
                            "pos": [471, 455],
                            "size": [315, 106],
                            "outputs": [
                                {
                                    "name": "LATENT",
                                    "type": "LATENT",
                                    "links": [51],
                                    "shape": 3,
                                    "slot_index": 0,
                                }
                            ],
                            "properties": {"Node name for S&R": "EmptySD3LatentImage"},
                            "widgets_values": [width, height, 1],
                            "color": "#323",
                            "bgcolor": "#535",
                        },
                        {
                            "id": 35,
                            "type": "FluxGuidance",
                            "pos": [576, 96],
                            "size": [211.6, 58],
                            "inputs": [
                                {"name": "conditioning", "type": "CONDITIONING", "link": 56}
                            ],
                            "outputs": [
                                {
                                    "name": "CONDITIONING",
                                    "type": "CONDITIONING",
                                    "links": [57],
                                    "shape": 3,
                                    "slot_index": 0,
                                }
                            ],
                            "properties": {"Node name for S&R": "FluxGuidance"},
                            "widgets_values": [3.5],
                        },
                        {
                            "id": 8,
                            "type": "VAEDecode",
                            "pos": [1151, 195],
                            "size": [210, 46],
                            "inputs": [
                                {"name": "samples", "type": "LATENT", "link": 52},
                                {"name": "vae", "type": "VAE", "link": 46},
                            ],
                            "outputs": [
                                {"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}
                            ],
                            "properties": {"Node name for S&R": "VAEDecode"},
                        },
                        {
                            "id": 30,
                            "type": "CheckpointLoaderSimple",
                            "pos": [48, 192],
                            "size": [315, 98],
                            "outputs": [
                                {
                                    "name": "MODEL",
                                    "type": "MODEL",
                                    "links": [47],
                                    "shape": 3,
                                    "slot_index": 0,
                                },
                                {
                                    "name": "CLIP",
                                    "type": "CLIP",
                                    "links": [45, 54],
                                    "shape": 3,
                                    "slot_index": 1,
                                },
                                {
                                    "name": "VAE",
                                    "type": "VAE",
                                    "links": [46],
                                    "shape": 3,
                                    "slot_index": 2,
                                },
                            ],
                            "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
                            "widgets_values": ["flux1-dev-fp8.safetensors"],
                        },
                    ],
                    "links": [
                        [9, 8, 0, 9, 0, "IMAGE"],
                        [45, 30, 1, 6, 0, "CLIP"],
                        [46, 30, 2, 8, 1, "VAE"],
                        [47, 30, 0, 31, 0, "MODEL"],
                        [51, 27, 0, 31, 3, "LATENT"],
                        [52, 31, 0, 8, 0, "LATENT"],
                        [54, 30, 1, 33, 0, "CLIP"],
                        [55, 33, 0, 31, 2, "CONDITIONING"],
                        [56, 6, 0, 35, 0, "CONDITIONING"],
                        [57, 35, 0, 31, 1, "CONDITIONING"],
                    ],
                    "extra": {
                        "ds": {"scale": 1.1, "offset": [-26.589059860274613, -54.534600602439575]}
                    },
                },
            }
        }

        return template

    def _poll_status(
        self, job_id: str, max_wait: int = 600, poll_interval: int = 2
    ) -> dict[str, Any]:
        """
        Poll the Runpod status endpoint until job completes or times out.

        Args:
            job_id: The Runpod job ID to poll
            max_wait: Maximum seconds to wait for completion
            poll_interval: Seconds between status checks

        Returns:
            The completed job response

        Raises:
            RuntimeError: If job fails or times out
        """
        # Construct status URL from run_url
        # Typical pattern: run_url ends with /run or /runsync
        # Status URL should be base_url/status/{job_id}
        base_url = self.run_url.rstrip("/")
        if base_url.endswith("/run") or base_url.endswith("/runsync"):
            base_url = base_url.rsplit("/", 1)[0]
        status_url = f"{base_url}/status/{job_id}"

        elapsed = 0
        while elapsed < max_wait:
            try:
                response = requests.get(
                    status_url,
                    headers=self.headers,
                    timeout=30,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to poll Runpod status: {exc}") from exc

            if response.status_code != 200:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise RuntimeError(f"Runpod status error {response.status_code}: {detail}")

            result = response.json()
            status = result.get("status")

            if status == "COMPLETED":
                return result
            elif status == "FAILED":
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"Runpod job failed: {error_msg}")
            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                # Still processing, wait and retry
                time.sleep(poll_interval)
                elapsed += poll_interval
            else:
                # Unknown status
                raise RuntimeError(f"Unexpected Runpod status: {status}")

        raise RuntimeError(f"Runpod job timed out after {max_wait} seconds")

    # ----------------------------------------------------------
    # public call
    # ----------------------------------------------------------
    def __call__(self, image_description: str, size: str, **kwargs: Any) -> tuple[str, float]:
        width, height = self._parse_size(size)

        payload = self._build_workflow(image_description, width, height)

        # Allow optional overrides via kwargs for advanced use-cases
        # e.g., seed, steps, cfg, scheduler, sampler_name, denoise, negative text, ckpt_name
        wf = payload["input"]["workflow"]
        sampler_inputs = wf["31"]["inputs"]
        if "seed" in kwargs:
            sampler_inputs["seed"] = int(kwargs["seed"])
        if "steps" in kwargs:
            sampler_inputs["steps"] = int(kwargs["steps"])
        if "cfg" in kwargs:
            sampler_inputs["cfg"] = float(kwargs["cfg"])
        if "sampler_name" in kwargs:
            sampler_inputs["sampler_name"] = str(kwargs["sampler_name"])
        if "scheduler" in kwargs:
            sampler_inputs["scheduler"] = str(kwargs["scheduler"])
        if "denoise" in kwargs:
            sampler_inputs["denoise"] = float(kwargs["denoise"])
        if "negative" in kwargs:
            wf["33"]["inputs"]["text"] = str(kwargs["negative"])
        if "ckpt_name" in kwargs:
            wf["30"]["inputs"]["ckpt_name"] = str(kwargs["ckpt_name"])
        if "batch_size" in kwargs:
            wf["27"]["inputs"]["batch_size"] = int(kwargs["batch_size"])

        # Submit the job
        try:
            response = requests.post(
                self.run_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30,
            )
        except Exception as exc:
            raise RuntimeError(f"Runpod request failed: {exc}") from exc

        if response.status_code != 200:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"Runpod error {response.status_code}: {detail}")

        out = response.json()

        # Check if this is an async endpoint (has job ID) or sync (immediate result)
        job_id = out.get("id")
        status = out.get("status")

        if job_id and status in ("IN_QUEUE", "IN_PROGRESS"):
            # Async endpoint - need to poll for completion
            out = self._poll_status(job_id, max_wait=kwargs.get("max_wait", 600))
        elif status is not None and status != "COMPLETED":
            raise RuntimeError(f"Runpod job did not complete successfully: {status}")

        # Extract the image from the completed response
        try:
            images = out["output"]["images"]
            base64_image = images[0]["data"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected Runpod response format: {out}") from exc

        cost = 0.0
        self.add_cost(cost)
        return base64_image, cost
