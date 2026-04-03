from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_zoo.rs_demo import cli
from model_zoo.rs_demo.config import parse_config, validate_torchrec_config


class TestTorchRecConfig(unittest.TestCase):
    def test_torchrec_backend_parses_profiler_flags(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-profiler",
                "--torchrec-profiler-warmup",
                "1",
                "--torchrec-profiler-active",
                "3",
                "--torchrec-profiler-repeat",
                "2",
                "--torchrec-trace-dir",
                "/tmp/example/trace",
                "--torchrec-main-csv",
                "/tmp/example/main.csv",
                "--torchrec-trace-csv",
                "/tmp/example/trace.csv",
            ]
        )
        self.assertEqual(cfg.backend, "torchrec")
        self.assertTrue(cfg.torchrec_profiler)
        self.assertEqual(cfg.torchrec_profiler_warmup, 1)
        self.assertEqual(cfg.torchrec_profiler_active, 3)
        self.assertEqual(cfg.torchrec_profiler_repeat, 2)
        self.assertEqual(cfg.torchrec_trace_dir, "/tmp/example/trace")
        self.assertEqual(cfg.torchrec_main_csv, "/tmp/example/main.csv")
        self.assertEqual(cfg.torchrec_trace_csv, "/tmp/example/trace.csv")

    def test_torchrec_no_start_server_flag(self) -> None:
        cfg = parse_config(["--backend", "torchrec", "--no-start-server"])
        self.assertEqual(cfg.backend, "torchrec")
        self.assertFalse(cfg.start_server)

    def test_torchrec_profiler_allows_subargs_when_enabled(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--torchrec-profiler",
                "--torchrec-profiler-warmup",
                "1",
                "--torchrec-profiler-active",
                "3",
                "--torchrec-profiler-repeat",
                "2",
                "--torchrec-trace-dir",
                "/tmp/example/trace",
                "--torchrec-trace-csv",
                "/tmp/example/trace.csv",
            ]
        )
        validate_torchrec_config(cfg)

    def test_torchrec_profiler_subargs_require_profiler_flag(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "TorchRec profiler sub-arguments require --torchrec-profiler"
        ):
            cfg = parse_config(
                [
                    "--backend",
                    "torchrec",
                    "--torchrec-profiler-warmup",
                    "1",
                ]
            )
            validate_torchrec_config(cfg)

    def test_cli_writes_trace_csv_only_when_profiler_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_root = Path(tmpdir) / "traces"
            trace_csv = Path(tmpdir) / "trace.csv"

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    trace_dir = Path(cfg.torchrec_trace_dir)
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    (trace_dir / "sample.pt.trace.json").write_text(
                        json.dumps(
                            {"traceEvents": [{"name": "cudaStreamSynchronize", "dur": 1000}]}
                        ),
                        encoding="utf-8",
                    )
                    return {"backend": "torchrec", "rows": []}

            with mock.patch.object(cli, "build_runner", return_value=_FakeRunner()):
                rc = cli.main(
                    [
                        "--backend",
                        "torchrec",
                        "--steps",
                        "1",
                        "--no-start-server",
                        "--torchrec-profiler",
                        "--torchrec-trace-dir",
                        str(trace_root),
                        "--torchrec-trace-csv",
                        str(trace_csv),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertTrue(trace_csv.exists())

    def test_cli_does_not_write_trace_csv_when_profiler_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            default_trace_csv = Path("/tmp/rs_demo_torchrec_trace.csv")
            if default_trace_csv.exists():
                default_trace_csv.unlink()

            class _FakeRunner:
                def run(self, repo_root, cfg):
                    return {"backend": "torchrec", "rows": []}

            with mock.patch.object(cli, "build_runner", return_value=_FakeRunner()):
                rc = cli.main(
                    [
                        "--backend",
                        "torchrec",
                        "--steps",
                        "1",
                        "--no-start-server",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertFalse(default_trace_csv.exists())


if __name__ == "__main__":
    unittest.main()
