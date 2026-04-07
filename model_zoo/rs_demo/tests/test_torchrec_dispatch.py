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

    def test_runner_builds_multi_node_torchrun_command(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=2,
            nnodes=2,
            node_rank=1,
            nproc=4,
            nproc_per_node=4,
            master_addr="10.0.2.191",
            master_port=29600,
            rdzv_backend="c10d",
            rdzv_id="demo-run",
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="case-a",
            torchrec_main_csv="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_main.csv",
            torchrec_main_agg_csv="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_main_agg.csv",
            torchrec_trace_dir="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_traces",
            torchrec_trace_csv="/nas/home/shq/docker/rs_demo/outputs/case-a/torchrec_trace.csv",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)
        self.assertIn("--nnodes", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--node_rank", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--nproc_per_node", cmd)
        self.assertIn("4", cmd)
        self.assertIn("--master_addr", cmd)
        self.assertIn("10.0.2.191", cmd)
        self.assertIn("--master_port", cmd)
        self.assertIn("29600", cmd)
        self.assertIn("--rdzv_backend", cmd)
        self.assertIn("c10d", cmd)
        self.assertIn("--rdzv_id", cmd)
        self.assertIn("demo-run", cmd)
        self.assertIn("--tee", cmd)
        self.assertIn("3", cmd)
        self.assertNotIn("--standalone", cmd)
        self.assertIn("--master-addr", cmd)
        self.assertIn("--master-port", cmd)
        self.assertIn("--rdzv-backend", cmd)
        self.assertIn("--rdzv-id", cmd)
        self.assertIn("--output-root", cmd)
        self.assertIn("--run-id", cmd)

    def test_runner_rank_output_dir_uses_shared_output_root(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nnodes=2,
            nproc_per_node=2,
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="case-b",
            torchrec_main_csv="/nas/home/shq/docker/rs_demo/outputs/case-b/torchrec_main.csv",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        rank_dir = runner._rank_output_dir(cfg)
        self.assertEqual(
            rank_dir,
            Path("/nas/home/shq/docker/rs_demo/outputs/case-b/torchrec_ranks"),
        )

    def test_runner_uses_world_size_from_nnodes_and_nproc_per_node(self) -> None:
        cfg = RunConfig(
            backend="torchrec",
            steps=1,
            nproc=1,
            nnodes=2,
            nproc_per_node=2,
            output_root="/nas/home/shq/docker/rs_demo",
            run_id="case-c",
            torchrec_main_csv="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_main.csv",
            torchrec_main_agg_csv="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_main_agg.csv",
            torchrec_trace_dir="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_traces",
            torchrec_trace_csv="/nas/home/shq/docker/rs_demo/outputs/case-c/torchrec_trace.csv",
        )
        runner = TorchRecRunner(Path("/tmp/runtime"))
        with mock.patch.object(runner, "_run_distributed", return_value={"backend": "torchrec", "rows": []}) as dist_run:
            result = runner.run(Path("/app/RecStore"), cfg)
        self.assertEqual(result["backend"], "torchrec")
        dist_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
