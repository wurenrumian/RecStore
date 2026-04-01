from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from ..config import RunConfig
from ..data.dlrm_source import (
    build_kjt_batch_from_dense_sparse_labels,
    build_table_offsets_from_eb_configs,
    build_train_dataloader,
    convert_kjt_ids_to_fused_ids,
    inject_project_paths,
)
from ..runtime.report import summarize_us
from .base import BenchmarkRunner


def detect_library_path(repo_root: Path, user_path: str) -> Path:
    if user_path:
        p = Path(user_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"library_path not found: {p}")
        return p

    candidates = [
        repo_root / "build/lib/lib_recstore_ops.so",
        repo_root / "build/lib/_recstore_ops.so",
        repo_root / "build/_recstore_ops.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find RecStore ops library. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


class RecStoreRunner(BenchmarkRunner):
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def run(self, repo_root: Path, cfg: RunConfig) -> dict:
        inject_project_paths(repo_root)
        from torchrec.datasets.criteo import DEFAULT_CAT_NAMES  # type: ignore
        from client import RecstoreClient  # type: ignore

        library_path = detect_library_path(repo_root, cfg.library_path)
        print(f"[rs_demo] repo_root={repo_root}")
        print(f"[rs_demo] backend=recstore")
        print(f"[rs_demo] library={library_path}")

        orig_cwd = Path.cwd()
        try:
            os.chdir(str(self.runtime_dir))
            torch.manual_seed(cfg.seed)
            client = RecstoreClient(library_path=str(library_path))

            dataset, dataloader = build_train_dataloader(
                repo_root=repo_root,
                data_dir_rel=cfg.data_dir,
                train_ratio=cfg.train_ratio,
                num_embeddings=cfg.num_embeddings,
                batch_size=cfg.batch_size,
            )

            eb_configs = [
                {
                    "name": f"t_{feature_name}",
                    "num_embeddings": int(cfg.num_embeddings),
                    "embedding_dim": int(cfg.embedding_dim),
                    "feature_names": [feature_name],
                }
                for feature_name in DEFAULT_CAT_NAMES
            ]

            t0 = time.perf_counter()
            for table_cfg in eb_configs:
                ok = client.init_embedding_table(
                    table_cfg["name"],
                    table_cfg["num_embeddings"],
                    table_cfg["embedding_dim"],
                )
                if not ok:
                    raise RuntimeError(f"init_embedding_table failed: {table_cfg['name']}")
            print(f"[rs_demo] init {len(eb_configs)} tables done in {(time.perf_counter() - t0):.3f}s")

            table_offsets = build_table_offsets_from_eb_configs(eb_configs, cfg.fuse_k)
            init_rows = min(int(cfg.init_rows), len(dataset))
            init_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=min(cfg.batch_size, init_rows),
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                num_workers=0,
            )

            init_written = 0
            t0 = time.perf_counter()
            known_fused_ids: set[int] = set()
            for dense_batch, sparse_batch, labels_batch in init_loader:
                _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                    dense_batch, sparse_batch, labels_batch
                )
                fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
                if fused_ids.numel() == 0:
                    continue
                vals = torch.randn(fused_ids.numel(), cfg.embedding_dim, dtype=torch.float32) * 0.01
                client.emb_write(fused_ids, vals)
                known_fused_ids.update(fused_ids.tolist())
                init_written += fused_ids.numel()
                if init_written >= init_rows * 26:
                    break
            print(
                f"[rs_demo] initial emb_write fused_rows={init_written} in {(time.perf_counter() - t0):.3f}s"
            )

            read_lat_us: list[float] = []
            update_lat_us: list[float] = []
            data_iter = iter(dataloader)
            for step in range(cfg.steps):
                try:
                    dense_batch, sparse_batch, labels_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    dense_batch, sparse_batch, labels_batch = next(data_iter)

                _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                    dense_batch, sparse_batch, labels_batch
                )
                fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
                if fused_ids.numel() == 0:
                    continue

                read_ids = fused_ids.contiguous()
                update_ids = fused_ids.clone().contiguous()
                grads = torch.randn(update_ids.numel(), cfg.embedding_dim, dtype=torch.float32) * 0.001

                # Ensure read-before-update does not hit missing keys and crash.
                if cfg.read_before_update:
                    cur_ids = read_ids.tolist()
                    missing_ids = [x for x in cur_ids if x not in known_fused_ids]
                    if missing_ids:
                        warm_ids = torch.tensor(missing_ids, dtype=torch.int64)
                        warm_vals = torch.zeros(len(missing_ids), cfg.embedding_dim, dtype=torch.float32)
                        client.emb_write(warm_ids, warm_vals)
                        known_fused_ids.update(missing_ids)

                if cfg.read_before_update:
                    t1 = time.perf_counter()
                    if cfg.read_mode == "direct":
                        _ = client.emb_read(read_ids, cfg.embedding_dim)
                    else:
                        pid = client.emb_prefetch(read_ids)
                        _ = client.emb_wait_result(pid, cfg.embedding_dim)
                    t2 = time.perf_counter()
                    if step >= cfg.warmup_steps:
                        read_lat_us.append((t2 - t1) * 1e6)

                t1 = time.perf_counter()
                client.emb_update_table("t_cat_0", update_ids, grads)
                t2 = time.perf_counter()
                if step >= cfg.warmup_steps:
                    update_lat_us.append((t2 - t1) * 1e6)
                known_fused_ids.update(update_ids.tolist())

                if (step + 1) % 10 == 0:
                    print(
                        f"[rs_demo] step {step + 1}/{cfg.steps} "
                        f"read={read_lat_us[-1] if read_lat_us else 0:.1f}us "
                        f"update={update_lat_us[-1] if update_lat_us else 0:.1f}us"
                    )

            print("[rs_demo] workload finished")
            print(f"[rs_demo] emb_read latency: {summarize_us(read_lat_us)}")
            print(f"[rs_demo] emb_update latency: {summarize_us(update_lat_us)}")
            return {
                "backend": "recstore",
                "read_lat_us": read_lat_us,
                "update_lat_us": update_lat_us,
            }
        finally:
            os.chdir(str(orig_cwd))
