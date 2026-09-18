import csv

import pytest

from openhail.utils.logging_utilities import (
    CsvMetricsLogger,
)


def test_csv_metrics_logger_uses_stable_schema(tmp_path):
    output = tmp_path / "metrics.csv"
    with CsvMetricsLogger(output, ("episode", "reward")) as logger:
        logger.log({"episode": 1, "reward": 12.5})

    with output.open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file)) == [{"episode": "1", "reward": "12.5"}]


def test_csv_metrics_logger_rejects_unknown_fields(tmp_path):
    with CsvMetricsLogger(tmp_path / "metrics.csv", ("episode",)) as logger:
        with pytest.raises(ValueError, match="Unexpected metric fields"):
            logger.log({"episode": 1, "unknown": 2})
