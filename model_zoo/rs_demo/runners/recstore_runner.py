from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ..config import RunConfig
from ..data.dlrm_source import (
    build_kjt_batch_from_dense_sparse_labels,
    build_table_offsets_from_eb_configs,
    build_train_dataloader,
    convert_kjt_ids_to_fused_ids,
    get_default_cat_names,
    inject_project_paths,
)
from ..runtime.aligned_training import (
    build_dense_stack,
    prepare_dense_input,
    run_dense_backward,
    sync_device,
)
from ..runtime.recstore_distributed import ShardedRecstoreClient
from ..runtime.report import finalize_recstore_row, summarize_us, write_stage_csv
from .base import BenchmarkRunner


@contextmanager
def stage_timer(row: dict[str, Any], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        row[key] = (time.perf_counter() - start) * 1e3


def _bool_int(flag: bool) -> int:
    return 1 if flag else 0


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
        from client import RecstoreClient  # type: ignore
        default_cat_names = get_default_cat_names()

        library_path = detect_library_path(repo_root, cfg.library_path)
        print(f"[rs_demo] repo_root={repo_root}")
        print(f"[rs_demo] backend=recstore")
        print(f"[rs_demo] library={library_path}")

        orig_cwd = Path.cwd()
        try:
            os.chdir(str(self.runtime_dir))
            torch.manual_seed(cfg.seed)
            raw_client = RecstoreClient(library_path=str(library_path))
            client = ShardedRecstoreClient(raw_client, self.runtime_dir)
            if cfg.read_before_update and cfg.read_mode == "prefetch":
                print("[rs_demo] sharded recstore path uses prefetch read mode")
            elif cfg.read_mode != "direct":
                print("[rs_demo] unknown read mode, fallback to direct read mode")

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
                for feature_name in default_cat_names
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

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dense_input_dim = cfg.embedding_dim * len(default_cat_names) + 13
            dense_module = build_dense_stack(torch, dense_input_dim).to(device)
            criterion = torch.nn.BCEWithLogitsLoss()
            dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)

            read_lat_us: list[float] = []
            update_lat_us: list[float] = []
            rows: list[dict[str, Any]] = []
            data_iter = iter(dataloader)
            for step in range(cfg.steps):
                row: dict[str, Any] = {
                    "backend": "recstore",
                    "nproc": 1,
                    "rank": 0,
                    "batch_size": cfg.batch_size,
                    "step": step,
                    "warmup_excluded": _bool_int(step < cfg.warmup_steps),
                }
                step_start = time.perf_counter()
                try:
                    raw_dense_batch, sparse_batch, labels_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    raw_dense_batch, sparse_batch, labels_batch = next(data_iter)

                with stage_timer(row, "batch_prepare_ms"):
                    dense_batch = raw_dense_batch

                with stage_timer(row, "input_pack_ms"):
                    _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                        dense_batch, sparse_batch, labels_batch
                    )
                    fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
                if fused_ids.numel() == 0:
                    continue

                read_ids = fused_ids.contiguous()
                batch_rows = dense_batch.shape[0]

                # Ensure read-before-update does not hit missing keys and crash.
                if cfg.read_before_update:
                    cur_ids = read_ids.tolist()
                    missing_ids = [x for x in cur_ids if x not in known_fused_ids]
                    if missing_ids:
                        warm_ids = torch.tensor(missing_ids, dtype=torch.int64)
                        warm_vals = torch.zeros(len(missing_ids), cfg.embedding_dim, dtype=torch.float32)
                        client.emb_write(warm_ids, warm_vals)
                        known_fused_ids.update(missing_ids)

                embeddings = None
                with stage_timer(row, "embed_lookup_local_ms"):
                    if cfg.read_before_update and cfg.read_mode == "prefetch":
                        embeddings = client.emb_read_prefetch(read_ids, cfg.embedding_dim)
                    else:
                        embeddings = client.emb_read(read_ids, cfg.embedding_dim)
                if embeddings is None:
                    raise RuntimeError("recstore emb_read returned no embeddings")
                if step >= cfg.warmup_steps:
                    read_lat_us.append(row["embed_lookup_local_ms"] * 1e3)

                with stage_timer(row, "embed_pool_local_ms"):
                    pooled_chunks = []
                    for table_idx in range(len(default_cat_names)):
                        start = table_idx * batch_rows
                        end = start + batch_rows
                        pooled_chunks.append(embeddings[start:end])
                    pooled_cpu = torch.cat(pooled_chunks, dim=1)

                with stage_timer(row, "output_unpack_ms"):
                    dense_input, pooled, labels = prepare_dense_input(
                        dense_batch=dense_batch,
                        pooled_source=pooled_cpu,
                        labels_batch=labels_batch,
                        torch=torch,
                        device=device,
                    )

                with stage_timer(row, "dense_fwd_ms"):
                    sync_device(torch, device)
                    logits = dense_module(dense_input)
                    loss = criterion(logits, labels)
                    sync_device(torch, device)

                with stage_timer(row, "backward_ms"):
                    pooled_grad = run_dense_backward(
                        loss=loss,
                        pooled=pooled,
                        dense_module=dense_module,
                        torch=torch,
                        device=device,
                    )

                with stage_timer(row, "optimizer_ms"):
                    sync_device(torch, device)
                    dense_optimizer.step()
                    dense_optimizer.zero_grad(set_to_none=True)
                    sync_device(torch, device)

                with stage_timer(row, "sparse_update_ms"):
                    sync_device(torch, device)
                    grad_chunks = torch.chunk(pooled_grad.to("cpu"), len(default_cat_names), dim=1)
                    update_grads = torch.cat(grad_chunks, dim=0).contiguous()
                    client.emb_update_table("t_cat_0", read_ids, update_grads)
                    sync_device(torch, device)

                if step >= cfg.warmup_steps:
                    update_lat_us.append(row["sparse_update_ms"] * 1e3)
                known_fused_ids.update(read_ids.tolist())
                row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
                rows.append(finalize_recstore_row(row))

                if (step + 1) % 10 == 0:
                    print(
                        f"[rs_demo] step {step + 1}/{cfg.steps} "
                        f"emb={rows[-1]['emb_stage_ms'] if rows else 0:.2f}ms "
                        f"step={rows[-1]['step_total_ms'] if rows else 0:.2f}ms"
                    )

            print("[rs_demo] workload finished")
            print(f"[rs_demo] emb_read latency: {summarize_us(read_lat_us)}")
            print(f"[rs_demo] emb_update latency: {summarize_us(update_lat_us)}")
            write_stage_csv(Path(cfg.recstore_main_csv), rows)
            return {
                "backend": "recstore",
                "read_lat_us": read_lat_us,
                "update_lat_us": update_lat_us,
                "rows": rows,
            }
        finally:
            os.chdir(str(orig_cwd))
