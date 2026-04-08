from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners.recstore_runner import RecStoreRunner


class _DummyDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(17, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakeShardedClient:
    def __init__(self) -> None:
        self.emb_read_calls = 0
        self.emb_read_prefetch_calls = 0
        self.emb_prefetch_calls = 0
        self.emb_wait_result_calls = 0
        self._last_prefetch_keys = torch.empty((0,), dtype=torch.int64)

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        return True

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        return None

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_calls += 1
        raise AssertionError("prefetch read mode should not call emb_read")

    def emb_read_prefetch(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.emb_read_prefetch_calls += 1
        return torch.zeros((keys.numel(), embedding_dim), dtype=torch.float32)

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        self.emb_prefetch_calls += 1
        raise AssertionError("prefetch read mode should use stable emb_read_prefetch")

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        self.emb_wait_result_calls += 1
        raise AssertionError("prefetch read mode should use stable emb_read_prefetch")

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        return None


class TestRecStoreRunner(unittest.TestCase):
    def test_read_before_update_prefetch_mode_uses_prefetch_wait(self) -> None:
        runner_runtime = Path(tempfile.mkdtemp())
        repo_root = Path("/app/RecStore")
        cfg = RunConfig(
            steps=1,
            warmup_steps=0,
            init_rows=1,
            batch_size=1,
            embedding_dim=4,
            num_embeddings=16,
            read_before_update=True,
            read_mode="prefetch",
            recstore_main_csv=str(runner_runtime / "main.csv"),
        )

        dense = torch.zeros((1, 13), dtype=torch.float32)
        sparse = torch.zeros((1, 1), dtype=torch.int64)
        labels = torch.zeros((1, 1), dtype=torch.float32)
        dataset = [(dense, sparse, labels)]
        dataloader = [(dense, sparse, labels)]

        fake_client = _FakeShardedClient()
        fake_client_module = types.ModuleType("client")
        fake_client_module.RecstoreClient = lambda library_path=None: object()

        with mock.patch.dict("sys.modules", {"client": fake_client_module}):
            with mock.patch("model_zoo.rs_demo.runners.recstore_runner.inject_project_paths", lambda *_: None):
                with mock.patch(
                    "model_zoo.rs_demo.runners.recstore_runner.detect_library_path",
                    lambda *_: repo_root / "build/lib/lib_recstore_ops.so",
                ):
                    with mock.patch(
                        "model_zoo.rs_demo.runners.recstore_runner.ShardedRecstoreClient",
                        lambda raw_client, runtime_dir: fake_client,
                    ):
                        with mock.patch(
                            "model_zoo.rs_demo.runners.recstore_runner.get_default_cat_names",
                            lambda: ["cat_0"],
                        ):
                            with mock.patch(
                                "model_zoo.rs_demo.runners.recstore_runner.build_train_dataloader",
                                lambda **kwargs: (dataset, dataloader),
                            ):
                                with mock.patch(
                                    "model_zoo.rs_demo.runners.recstore_runner.build_table_offsets_from_eb_configs",
                                    lambda *args, **kwargs: {},
                                ):
                                    with mock.patch(
                                        "model_zoo.rs_demo.runners.recstore_runner.build_kjt_batch_from_dense_sparse_labels",
                                        lambda *args, **kwargs: (None, object()),
                                    ):
                                        with mock.patch(
                                            "model_zoo.rs_demo.runners.recstore_runner.convert_kjt_ids_to_fused_ids",
                                            lambda *args, **kwargs: torch.tensor([3], dtype=torch.int64),
                                        ):
                                            with mock.patch(
                                                "model_zoo.rs_demo.runners.recstore_runner.build_dense_stack",
                                                lambda *args, **kwargs: _DummyDense(),
                                            ):
                                                with mock.patch(
                                                    "model_zoo.rs_demo.runners.recstore_runner.prepare_dense_input",
                                                    lambda **kwargs: (
                                                        torch.zeros((1, 17), dtype=torch.float32, device=kwargs["device"]),
                                                        torch.zeros((1, 4), dtype=torch.float32, device=kwargs["device"], requires_grad=True),
                                                        torch.zeros((1, 1), dtype=torch.float32, device=kwargs["device"]),
                                                    ),
                                                ):
                                                    with mock.patch(
                                                        "model_zoo.rs_demo.runners.recstore_runner.run_dense_backward",
                                                        lambda **kwargs: torch.zeros((1, 4), dtype=torch.float32),
                                                    ):
                                                        with mock.patch(
                                                            "model_zoo.rs_demo.runners.recstore_runner.sync_device",
                                                            lambda *args, **kwargs: None,
                                                        ):
                                                            with mock.patch(
                                                                "model_zoo.rs_demo.runners.recstore_runner.finalize_recstore_row",
                                                                lambda row: row,
                                                            ):
                                                                with mock.patch(
                                                                    "model_zoo.rs_demo.runners.recstore_runner.summarize_us",
                                                                    lambda xs: "ok",
                                                                ):
                                                                    with mock.patch(
                                                                        "model_zoo.rs_demo.runners.recstore_runner.write_stage_csv",
                                                                        lambda *args, **kwargs: None,
                                                                    ):
                                                                        runner = RecStoreRunner(runner_runtime)
                                                                        runner.run(repo_root=repo_root, cfg=cfg)

        self.assertEqual(fake_client.emb_read_calls, 0)
        self.assertEqual(fake_client.emb_read_prefetch_calls, 1)
        self.assertEqual(fake_client.emb_prefetch_calls, 0)
        self.assertEqual(fake_client.emb_wait_result_calls, 0)


if __name__ == "__main__":
    unittest.main()
