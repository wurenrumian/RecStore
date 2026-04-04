from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.runtime.torchrec_compare import (
    build_compare_rows,
    write_compare_csv,
)


class TestTorchRecCompare(unittest.TestCase):
    def test_build_compare_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            with recstore_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "network_transport_us",
                        "storage_backend_update_us",
                        "server_total_us",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "network_transport_us": 2000,
                        "storage_backend_update_us": 3000,
                        "server_total_us": 4000,
                    }
                )

            with torchrec_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "collective_total_ms",
                        "kv_local_only_ms",
                        "kv_extended_ms",
                        "network_proxy_torchrec_extended_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "collective_total_ms": 1.0,
                        "kv_local_only_ms": 2.0,
                        "kv_extended_ms": 3.0,
                        "network_proxy_torchrec_extended_ms": 1.5,
                    }
                )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["network_main"]["recstore_ms"], 2.0)
        self.assertEqual(by_metric["network_main"]["torchrec_ms"], 1.0)
        self.assertEqual(by_metric["kv_strict"]["recstore_ms"], 3.0)
        self.assertEqual(by_metric["kv_strict"]["torchrec_ms"], 2.0)

    def test_write_compare_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "compare.csv"
            write_compare_csv(
                out_path,
                [
                    {
                        "metric": "network_main",
                        "recstore_ms": 2.0,
                        "torchrec_ms": 1.0,
                        "delta_ms": 1.0,
                        "delta_ratio": 1.0,
                    }
                ],
            )

            with out_path.open("r", encoding="utf-8") as f:
                row = next(csv.DictReader(f))

        self.assertEqual(row["metric"], "network_main")
        self.assertEqual(row["delta_ms"], "1.0")


if __name__ == "__main__":
    unittest.main()
