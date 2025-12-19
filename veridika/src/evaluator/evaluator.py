import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import threading
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from veridika.src.web_search.default_ban_domains import ban_domains
from veridika.src.workflows import Workflow

load_dotenv(".env")


class Evaluator:
    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        """Initialize the Evaluator with configuration and setup.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary. Defaults to None.
            config_path (Optional[str]): Path to configuration file. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If neither config nor config_path is provided, or if output_dir is not specified.
        """
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config or config_path must be provided")

        self._load_dataset()

        self.config["workflow_config"]["web_search"]["ban_domains"] = ban_domains + list(
            self.dataset_domains
        )
        self.config["workflow_config"]["web_search"]["ban_domains"] = [
            x for x in self.config["workflow_config"]["web_search"]["ban_domains"] if len(x) > 0
        ]

        if not self.config.get("output_dir"):
            raise ValueError(
                "Output directory is not specified in the config file. Please specify an output directory in the config file using the 'output_dir' key."
            )

        self.output_dir = Path(self.config.get("output_dir"))
        if self.config.get("overwrite_output_dir"):
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_arguments = self.config.get("run_arguments", {})
        self.processed_ids = set()
        self.total_cost = 0.0
        self.start_date = datetime.now().isoformat()
        self.prev_config = None

        # Thread synchronization lock for shared state
        self._lock = threading.Lock()

        self._restore_progress()

    def _load_dataset(self):
        """Load the FactCheckingEval dataset and extract metadata.

        Args:
            None

        Returns:
            None
        """
        self.dataset = load_dataset("Iker/FactCheckingEval")
        self.dataset_ids = set()
        self.dataset_domains = set()
        self.dataset_splits = set()
        for split in self.dataset:
            self.dataset_splits.add(split)
            for example in self.dataset[split]:
                self.dataset_ids.add(example["id"])
                self.dataset_domains.add(example["source_name"])

        logging.info(f"Loaded {len(self.dataset_ids)} examples. Splits: {self.dataset_splits}")
        logging.info(f"Banned domains: {self.dataset_domains}")

    def get_workflow(self):
        """Load and configure the workflow with banned domains.

        Args:
            None

        Returns:
            None
        """
        return Workflow(config=self.config["workflow_config"])

    def _load_config(self, config_path: str) -> dict[str, Any]:
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
        config = OmegaConf.to_container(omega_config, resolve=True)

        # If workflow_config is a string path, load it as a nested config
        if "workflow_config" in config and isinstance(config["workflow_config"], str):
            workflow_config_path = Path(config["workflow_config"])

            if workflow_config_path.exists():
                workflow_omega_config = OmegaConf.load(workflow_config_path)
                config["workflow_config"] = OmegaConf.to_container(
                    workflow_omega_config, resolve=True
                )
            else:
                raise FileNotFoundError(
                    f"Workflow configuration file not found: {workflow_config_path}"
                )

        return config

    def _configs_are_equal(self, config1: dict[str, Any], config2: dict[str, Any]) -> bool:
        """
        Compare two configurations for equality, ignoring certain fields that may change between runs.

        Args:
            config1: First configuration to compare
            config2: Second configuration to compare

        Returns:
            bool: True if configurations are equivalent for evaluation purposes
        """
        # Create copies to avoid modifying original configs
        config1["workflow_config"]["web_search"]["ban_domains"].sort()
        config2["workflow_config"]["web_search"]["ban_domains"].sort()

        config1_copy = json.loads(json.dumps(config1))
        config2_copy = json.loads(json.dumps(config2))

        # Remove fields that are expected to change between runs
        fields_to_ignore = ["output_dir", "overwrite_output_dir"]

        for field in fields_to_ignore:
            config1_copy.pop(field, None)
            config2_copy.pop(field, None)

        return config1_copy == config2_copy

    def _restore_progress(self):
        """Restore the progress of the evaluator from the output directory.

        The output directory should have the following structure:
        - output_dir/
            - evaluation_summary.json
            - configuration.json
            - example_id.json
            - example_id.json
            - ...

        The evaluation_summary file should contain the following fields:
        - total_cost: float
        - start_date: str
        - end_date: str

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If configuration mismatch is detected between current and previous runs.
        """
        if not self.output_dir.exists():
            return

        configuration_file = self.output_dir / "configuration.json"

        # Check if previous configuration exists
        if configuration_file.exists():
            with open(configuration_file) as f:
                self.prev_config = json.load(f)

            # Compare previous config with current config
            if not self._configs_are_equal(self.prev_config, self.config):
                raise ValueError(
                    "Configuration mismatch detected. The current configuration is different from the "
                    "previous configuration used for this evaluation run. Restoring progress is not "
                    "possible with a different configuration. Please use the same configuration or "
                    "start a new evaluation with 'overwrite_output_dir: true' in your config."
                )
        else:
            # First run - write the configuration file
            with open(configuration_file, "w") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

        evaluation_summary_file = self.output_dir / "evaluation_summary.json"
        if evaluation_summary_file.exists():
            with open(evaluation_summary_file) as f:
                evaluation_summary = json.load(f)
                self.total_cost = evaluation_summary["total_cost"]
                self.start_date = evaluation_summary["start_date"]

        for file in self.output_dir.glob("*.json"):
            if file.exists() and file.stem in self.dataset_ids:
                self.processed_ids.add(file.stem)

    def _save_result(self, result_id: str, result: dict[str, Any], cost: float):
        """Save the result of the evaluator to the output directory.

        The output directory should have the following structure:
        - output_dir/
            - example_id.json
            - example_id.json
            - ...

        Args:
            result_id (str): Unique identifier for the result
            result (Dict[str, Any]): Result data to save
            cost (float): Cost associated with processing this result

        Returns:
            None
        """
        result_file = self.output_dir / f"{result_id}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        # Thread-safe update of total cost
        with self._lock:
            self.total_cost += cost
            current_total_cost = self.total_cost

        evaluation_summary = {
            "total_cost": current_total_cost,
            "start_date": self.start_date,
            "end_date": None,  # Will be set when the evaluation is finished
        }
        with open(self.output_dir / "evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4, ensure_ascii=False)

    def _progress_bar_text(self):
        """Return the progress bar text with cost information.

        Args:
            None

        Returns:
            str: Formatted progress bar text showing current and estimated total cost
        """
        # Thread-safe reading of shared variables
        with self._lock:
            total_cost = self.total_cost
            processed_count = len(self.processed_ids)

        if processed_count == 0:
            estimated_total_cost_str = "TBD"
        else:
            estimated_total_cost = total_cost / processed_count * len(self.dataset_ids)
            estimated_total_cost_str = f"${estimated_total_cost:.4f}"

        return f"Cost: ${total_cost:.4f} / {estimated_total_cost_str}"

    def run(self, example: dict[str, Any]):
        """Run the evaluator on a single example.

        Args:
            example (Dict[str, Any]): Example data containing statement and metadata

        Returns:
            None

        Raises:
            ValueError: If the example is missing required fields
            Exception: For any workflow execution errors
        """
        # Validate example structure
        if "statement" not in example:
            raise ValueError(f"Example {example.get('id', 'unknown')} is missing 'statement' field")

        if "id" not in example:
            raise ValueError("Example is missing 'id' field")

        statement = example["statement"]
        example_id = example["id"]

        workflow = self.get_workflow()

        try:
            # Log the start of processing
            logging.info(f"Starting processing for example {example_id}: {statement[:100]}...")

            result, cost = asyncio.run(workflow.run(statement=statement, **self.run_arguments))

            # Validate that result has required fields
            if not isinstance(result, dict):
                raise ValueError(f"Workflow returned invalid result type: {type(result)}")

            if "metadata" not in result:
                raise ValueError(f"Workflow result missing metadata field for example {example_id}")

            if "label" not in result.get("metadata", {}):
                raise ValueError(
                    f"Workflow result missing metadata.label field for example {example_id}"
                )

            self._save_result(example_id, result, cost)
            logging.info(f"Successfully processed example {example_id} with cost ${cost:.4f}")

        except Exception as e:
            # Add more context to the error
            error_msg = f"Failed to process example {example_id} (statement: '{statement[:100]}...'): {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def __call__(self, max_workers: int = 4):
        """Run the evaluator on all unprocessed examples in parallel.

        Args:
            max_workers (int): Maximum number of parallel workers for processing examples. Defaults to 4.

        Returns:
            None
        """
        # Collect all unprocessed examples
        unprocessed_examples = []
        for split in self.dataset_splits:
            for example in self.dataset[split]:
                if example["id"] not in self.processed_ids:
                    unprocessed_examples.append(example)

        if not unprocessed_examples:
            logging.info("All examples have already been processed.")
            return

        logging.info(
            f"Processing {len(unprocessed_examples)} unprocessed examples with {max_workers} workers..."
        )

        # Initialize progress bar
        pbar = tqdm(
            total=len(unprocessed_examples),
            desc=self._progress_bar_text() if self.processed_ids else "Cost: $0.0000 / $0.0000",
            unit="examples",
        )

        def process_example_wrapper(example):
            """Wrapper function to handle exceptions during processing"""
            try:
                self.run(example)
                return example["id"], None
            except Exception as e:
                logging.exception(f"Error processing example {example['id']}: {str(e)}")
                return example["id"], str(e)

        # Process examples in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(process_example_wrapper, example): example
                for example in unprocessed_examples
            }

            # Process completed tasks
            completed_count = 0
            failed_examples = []

            for future in as_completed(future_to_example):
                example_id, error = future.result()
                completed_count += 1

                if error:
                    failed_examples.append((example_id, error))
                else:
                    # Thread-safe update of processed_ids for successful examples
                    with self._lock:
                        self.processed_ids.add(example_id)

                # Update progress bar
                pbar.update(1)
                pbar.set_description(self._progress_bar_text())

        pbar.close()

        # Report results
        successful_count = completed_count - len(failed_examples)
        logging.info(
            f"Completed processing: {successful_count} successful, {len(failed_examples)} failed"
        )

        if failed_examples:
            logging.warning("Failed examples:")
            for example_id, error in failed_examples:
                logging.warning(f"  {example_id}: {error}")

        else:
            self.evaluate()

    def evaluate(self):
        """Evaluate the results when the evaluation is finished.

        Calculates performance metrics (accuracy, precision, recall, F1-score)
        for each dataset split and overall results.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If result file not found for an example or invalid label encountered.
        """

        splits_results = {}
        statement_results = {}

        for split in self.dataset_splits:
            if split not in splits_results:
                splits_results[split] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "Undetermined": 0}

            for example in self.dataset[split]:
                example_id = example["id"]
                example_label = example["label"]
                result_file = self.output_dir / f"{example_id}.json"
                if result_file.exists():
                    with open(result_file) as f:
                        result = json.load(f)
                        result_label = result["metadata"]["label"]
                else:
                    raise ValueError(
                        f"Result file not found for example {example_id}. Evaluator has not been run for this example."
                    )

                if example_label is True:  # Actual = True
                    if result_label == "True":
                        splits_results[split]["TP"] += 1  # True Positive
                    elif result_label == "Fake":
                        splits_results[split]["FN"] += 1  # False Negative
                    elif result_label == "Undetermined":
                        splits_results[split]["Undetermined"] += 1
                    else:
                        raise ValueError(f"Invalid label: {result_label}")
                else:  # Actual = False
                    if result_label == "True":  # Fixed: was checking "is False"
                        splits_results[split]["FP"] += 1  # False Positive
                    elif result_label == "Fake":
                        splits_results[split]["TN"] += 1  # True Negative
                    elif result_label == "Undetermined":
                        splits_results[split]["Undetermined"] += 1
                    else:
                        raise ValueError(f"Invalid label: {result_label}")

                statement_results[example_id] = {
                    "id": example_id,
                    "split": split,
                    "statement": example["statement"],
                    "gold_label": example_label,
                    "predicted_label": result_label,
                }

        full_results = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "Undetermined": 0}

        # Helper function to calculate metrics with zero division protection and scaling
        def calculate_metrics(tp, tn, fp, fn, undetermined):
            total_determined = tp + tn + fp + fn
            total_all = total_determined + undetermined

            metrics = {}

            # Accuracy: (TP + TN) / (TP + TN + FP + FN) - excluding undetermined
            metrics["accuracy"] = (
                round((tp + tn) / total_determined * 100, 2) if total_determined > 0 else 0.0
            )

            # Precision: TP / (TP + FP)
            metrics["precision"] = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0

            # Recall: TP / (TP + FN)
            metrics["recall"] = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0

            # F1 Score: 2 * (precision * recall) / (precision + recall)
            precision_decimal = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_decimal = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision_decimal + recall_decimal > 0:
                metrics["f1_score"] = round(
                    2
                    * precision_decimal
                    * recall_decimal
                    / (precision_decimal + recall_decimal)
                    * 100,
                    2,
                )
            else:
                metrics["f1_score"] = 0.0

            # Undetermined Rate: Undetermined / Total
            metrics["undetermined_rate"] = (
                round(undetermined / total_all * 100, 2) if total_all > 0 else 0.0
            )

            return metrics

        # Calculate metrics for each split
        for split in splits_results:
            tp = splits_results[split]["TP"]
            tn = splits_results[split]["TN"]
            fp = splits_results[split]["FP"]
            fn = splits_results[split]["FN"]
            undetermined = splits_results[split]["Undetermined"]

            full_results["TP"] += tp
            full_results["FP"] += fp
            full_results["FN"] += fn
            full_results["TN"] += tn
            full_results["Undetermined"] += undetermined

            # Add calculated metrics to split results
            splits_results[split].update(calculate_metrics(tp, tn, fp, fn, undetermined))

        # Calculate metrics for full results
        full_results.update(
            calculate_metrics(
                full_results["TP"],
                full_results["TN"],
                full_results["FP"],
                full_results["FN"],
                full_results["Undetermined"],
            )
        )

        with open(self.output_dir / "evaluation_summary.json") as f:
            evaluation_summary = json.load(f)

        with open(self.output_dir / "evaluation_summary.json", "w") as f:
            evaluation_summary["end_date"] = datetime.now().isoformat()
            evaluation_summary["results"] = {
                "splits": splits_results,
                "full": full_results,
            }
            json.dump(evaluation_summary, f, indent=4, ensure_ascii=False)

        with open(self.output_dir / "statement_results.json", "w") as f:
            json.dump(statement_results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    evaluator = Evaluator(config_path=args.config)
    evaluator(max_workers=args.max_workers)
