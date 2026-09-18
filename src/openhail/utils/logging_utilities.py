import csv
import dataclasses
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Mapping, Sequence

logger_blocklist = [
    "fiona",
    "rasterio",
    "matplotlib",
    "PIL",
    "geopandas",
]


# Required schema for the forthcoming actor-critic trainer. Evaluation rows
# leave optimization-only fields empty but retain the same stable columns.
TRAINING_METRIC_FIELDS = (
    "episode",
    "phase",
    "seed",
    "policy_seed",
    "action_selection",
    "reward",
    "service_reward",
    "reposition_reward",
    "requests_total",
    "requests_served",
    "request_acceptance_rate",
    "action_noop_count",
    "action_reposition_count",
    "action_charge_count",
    "zero_capacity_charge_count",
    "reposition_destination_counts",
    "charge_destination_counts",
    "policy_decision_count",
    "mean_policy_delta_time",
    "policy_entropy",
    "policy_loss",
    "approximate_kl",
    "clip_fraction",
    "value_loss",
    "explained_variance",
    "gradient_norm",
    "policy_gradient_norm",
    "value_gradient_norm",
    "optimizer_updates",
    "learning_rate",
    "critic_learning_rate",
    "invalid_action_count",
    "empty_mask_count",
    "mean_soc",
    "mean_charger_occupancy",
    "mean_charger_queue",
    "mean_delta_time",
    "steps",
)


def init_logging(
    log_name, inittime, stdout=False, level=logging.INFO, log_folder="./logging"
):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=f"{log_folder}/{inittime}.{log_name}.log", level=level)
    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logging.root.addHandler(stdout_handler)
    set_filters()


def init_logging_stdout(level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(stream=sys.stdout, level=level)
    set_filters()


def set_filters():
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*Box observation space.*"
    )
    # warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_charger_name(charger_config):
    if charger_config["mode"] == "infinite":
        return f"CI_{charger_config['locations']}"
    elif charger_config["mode"] == "uniform":
        return f"CU_{charger_config['locations']}_{charger_config['count']}"
    elif charger_config["mode"] == "limited":
        return f"CV_{charger_config['locations']}_{charger_config['count']}"
    elif charger_config["mode"] == "select":
        return (
            f"CS_{charger_config['locations']}_{charger_config['selection']}_"
            f"{charger_config['count']}"
        )
    raise ValueError(f"Unrecognized sample mode {charger_config['mode']}")


def get_run_name(instance_config, extra_config, run_description=""):
    names = [
        instance_config["evaluation"]["name"],
        instance_config["city"]["name"],
        instance_config["horizon"]["name"],
        instance_config["fleet"]["name"],
        instance_config["infrastructure"]["name"],
        extra_config["agent"]["name"],
    ]
    if run_description:
        names.insert(0, run_description)
    return ".".join(names)


# ------------------------------------------------------------------------------


@dataclasses.dataclass
class OutputManager:
    filename: str
    existing: bool = False

    def __init__(self, filename):
        self.filename = f"./out/{filename}.txt"
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        self.existing = False
        if os.path.isfile(self.filename):
            logging.info(
                f"Output file set to: {self.filename} - existing file. Appending to it."
            )
            self.existing = True
        else:
            logging.info(f"Output file set to: {self.filename} - new file.")

    def log_entry(self, stats):
        with open(self.filename, mode="a+", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter="\t")
            writer.writerows([stats])


class CsvMetricsLogger:
    """Write flushed, schema-stable metric rows for reproducible runs."""

    def __init__(
        self,
        filename: str | os.PathLike,
        fieldnames: Sequence[str],
        append: bool = False,
    ) -> None:
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = tuple(fieldnames)
        mode = "a" if append else "w"
        write_header = (
            not append or not self.path.exists() or self.path.stat().st_size == 0
        )
        self._file = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, metrics: Mapping[str, object]) -> None:
        """Write one row, rejecting fields outside the declared schema."""
        extra = set(metrics) - set(self.fieldnames)
        if extra:
            raise ValueError(f"Unexpected metric fields: {sorted(extra)}")
        row = {field: metrics.get(field, "") for field in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
