from __future__ import annotations

from pathlib import Path
from unittest import mock
import unittest

from model_zoo.rs_demo.cli import build_runner
from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners.torchrec_runner import TorchRecRunner


class TestTorchRecDispatch(unittest.TestCase):
    def test_build_runner_torchrec_requires_dependency(self) -> None:
        cfg = RunConfig(backend="torchrec")
        with mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_torchrec_available",
            side_effect=RuntimeError(
                "TorchRec backend requires the `torchrec` package to be installed."
            ),
        ), self.assertRaisesRegex(
            RuntimeError, "TorchRec backend requires the `torchrec` package"
        ):
            build_runner(cfg, Path("/tmp"))

    def test_build_runner_torchrec_returns_runner_when_dependency_available(self) -> None:
        cfg = RunConfig(backend="torchrec")
        with mock.patch(
            "model_zoo.rs_demo.runners.torchrec_runner.ensure_torchrec_available",
            return_value=None,
        ):
            runner = build_runner(cfg, Path("/tmp"))
        self.assertEqual(runner.__class__.__name__, "TorchRecRunner")

    def test_runner_rejects_profiler_subargs_before_dependency_check(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            torchrec_profiler_warmup=1,
        )
        runner = TorchRecRunner(Path("/tmp"))
        with self.assertRaisesRegex(
            RuntimeError, "TorchRec profiler sub-arguments require --torchrec-profiler"
        ):
            runner.run(Path("/app/RecStore"), cfg)

    def test_runner_rejects_non_torchrec_backend(self) -> None:
        runner = TorchRecRunner(Path("/tmp"))
        with self.assertRaisesRegex(
            ValueError, "TorchRecRunner requires cfg.backend to be 'torchrec'"
        ):
            runner.run(Path("/app/RecStore"), RunConfig(backend="recstore", steps=1))


if __name__ == "__main__":
    unittest.main()
