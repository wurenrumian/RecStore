from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from ..config import RunConfig, validate_torchrec_config
from ..runtime.report import finalize_torchrec_row, write_stage_csv
from ..runtime.torchrec_profile import build_torchrec_profiler
from .base import BenchmarkRunner


def ensure_torchrec_available() -> None:
    try:
        import torchrec.datasets.criteo  # noqa: F401
        import torchrec.modules.embedding_configs  # noqa: F401
        import torchrec.modules.embedding_modules  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchRec backend requires the `torchrec` package to be installed."
        ) from exc


@contextmanager
def stage_timer(row: dict[str, Any], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        row[key] = (time.perf_counter() - start) * 1e3


class TorchRecRunner(BenchmarkRunner):
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def run(self, repo_root: Path, cfg: RunConfig) -> dict:
        if cfg.backend != "torchrec":
            raise ValueError("TorchRecRunner requires cfg.backend to be 'torchrec'.")
        validate_torchrec_config(cfg)
        if cfg.steps <= 0:
            raise ValueError("TorchRec runner requires --steps to be greater than 0.")

        ensure_torchrec_available()
        import torch
        from torch import nn

        from ..data.dlrm_source import (
            build_kjt_batch_from_dense_sparse_labels,
            build_train_dataloader,
            inject_project_paths,
        )

        inject_project_paths(repo_root)

        from torchrec.datasets.criteo import DEFAULT_CAT_NAMES
        from torchrec.modules.embedding_configs import EmbeddingBagConfig
        from torchrec.modules.embedding_modules import EmbeddingBagCollection

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)

        _dataset, dataloader = build_train_dataloader(
            repo_root=repo_root,
            data_dir_rel=cfg.data_dir,
            train_ratio=cfg.train_ratio,
            num_embeddings=cfg.num_embeddings,
            batch_size=cfg.batch_size,
        )
        data_iter = iter(dataloader)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=int(cfg.embedding_dim),
                num_embeddings=int(cfg.num_embeddings),
                feature_names=[feature_name],
            )
            for feature_name in DEFAULT_CAT_NAMES
        ]

        ebc = EmbeddingBagCollection(tables=eb_configs, device=device)
        dense = nn.Sequential(
            nn.Linear(cfg.embedding_dim * len(DEFAULT_CAT_NAMES) + 13, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            list(ebc.parameters()) + list(dense.parameters()), lr=0.01
        )

        rows = []
        profiler = build_torchrec_profiler(cfg)
        profiler_context = profiler or nullcontext()

        with profiler_context:
            for step in range(cfg.steps):
                row: dict[str, float | int | str] = {
                    "backend": "torchrec",
                    "batch_size": cfg.batch_size,
                    "step": step,
                    "warmup_excluded": int(step < cfg.warmup_steps),
                    "collective_mode": "not_measured_single_process",
                    "collective_measured": 0,
                }
                step_start = time.perf_counter()

                with stage_timer(row, "batch_prepare_ms"):
                    try:
                        dense_batch, sparse_batch, labels_batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        dense_batch, sparse_batch, labels_batch = next(data_iter)

                with stage_timer(row, "input_pack_ms"):
                    dense_batch, sparse_features = (
                        build_kjt_batch_from_dense_sparse_labels(
                            dense_batch, sparse_batch, labels_batch
                        )
                    )

                with stage_timer(row, "embed_lookup_local_ms"):
                    embeddings = ebc(sparse_features.to(device))

                with stage_timer(row, "embed_pool_local_ms"):
                    flat_embeddings = [embeddings[key] for key in DEFAULT_CAT_NAMES]
                    pooled = torch.cat(flat_embeddings, dim=1)

                # Single-process Task 2 runner does not issue collectives yet.
                row["collective_launch_ms"] = 0.0
                row["collective_wait_ms"] = 0.0

                with stage_timer(row, "output_unpack_ms"):
                    dense_input = torch.cat([dense_batch.to(device), pooled], dim=1)

                with stage_timer(row, "dense_fwd_ms"):
                    logits = dense(dense_input)
                    labels = labels_batch.to(device).float()
                    if labels.ndim == 1:
                        labels = labels.view(-1, 1)
                    loss = criterion(logits, labels)

                with stage_timer(row, "backward_ms"):
                    loss.backward()

                with stage_timer(row, "optimizer_ms"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if profiler is not None:
                    profiler.step()

                row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
                rows.append(finalize_torchrec_row(row))

                if (step + 1) % 10 == 0:
                    print(f"[rs_demo] torchrec step {step + 1}/{cfg.steps}")

        write_stage_csv(Path(cfg.torchrec_main_csv), rows)
        return {"backend": "torchrec", "rows": rows}
