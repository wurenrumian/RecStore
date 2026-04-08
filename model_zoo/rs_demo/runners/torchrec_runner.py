from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from ..config import RunConfig, ensure_shared_dir, validate_torchrec_config
from ..runtime.aligned_training import (
    build_dense_stack,
    prepare_dense_input,
    run_dense_backward,
    sync_device,
)
from ..runtime.report import finalize_torchrec_row, write_stage_csv
from ..runtime.torchrec_profile import build_torchrec_profiler
from .base import BenchmarkRunner


def ensure_torchrec_available() -> None:
    try:
        import torchrec.datasets.criteo  # noqa: F401
        import torchrec.distributed.model_parallel  # noqa: F401
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


def _bool_int(flag: bool) -> int:
    return 1 if flag else 0


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_stage_csv(path, rows)


def _pick_socket_ifname() -> str | None:
    preferred = ("eno1", "eno8303")
    try:
        available = set(os.listdir("/sys/class/net"))
    except OSError:
        return None
    for name in preferred:
        if name in available:
            return name
    return None


def _append_worker_debug(cfg: RunConfig, rank: int, message: str) -> None:
    debug_path = Path(cfg.output_root) / "outputs" / cfg.run_id / "torchrec_worker_debug.log"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} rank={rank} {message}\n")


def _merge_rank_outputs(paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in paths:
        for row in _load_rows(path):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = ""
                    continue
                if key in {"backend", "collective_mode"}:
                    normalized[key] = value
                    continue
                try:
                    if "." in value:
                        normalized[key] = float(value)
                    else:
                        normalized[key] = int(value)
                except (TypeError, ValueError):
                    normalized[key] = value
            merged.append(normalized)
    merged.sort(key=lambda row: (int(row.get("rank", 0)), int(row.get("step", 0))))
    _write_rows(out_path, merged)
    return merged


def _make_trace_handler(cfg: RunConfig, rank: int):
    def _handler(prof) -> None:
        trace_path = Path(cfg.torchrec_trace_dir) / f"rank{rank}.pt.trace.json"
        prof.export_chrome_trace(str(trace_path))

    return _handler


def _run_single_or_dist_worker(
    repo_root: Path,
    cfg: RunConfig,
    rank: int,
    world_size: int,
    local_rank: int,
    out_csv: Path,
) -> list[dict[str, Any]]:
    import torch
    from torch import distributed as dist
    from torch import nn

    from ..data.dlrm_source import (
        build_kjt_batch_from_dense_sparse_labels,
        build_train_dataloader,
        inject_project_paths,
    )

    inject_project_paths(repo_root)

    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES
    from torchrec.distributed.model_parallel import (
        DistributedModelParallel,
        get_default_sharders,
    )
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection

    is_dist = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    _append_worker_debug(
        cfg,
        rank,
        f"worker_start world_size={world_size} local_rank={local_rank} backend={backend}",
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if is_dist and not dist.is_initialized():
        _append_worker_debug(cfg, rank, f"before_init_process_group device={device}")
        dist.init_process_group(backend=backend, device_id=device if device.type == "cuda" else None)
        _append_worker_debug(cfg, rank, "after_init_process_group")

    torch.manual_seed(cfg.seed + rank)

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

    embedding_module = EmbeddingBagCollection(tables=eb_configs, device=device)
    use_dist = world_size > 1
    if use_dist:
        _append_worker_debug(cfg, rank, "before_sharding_plan")
        sharders = get_default_sharders()
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size,
                local_world_size=cfg.nproc_per_node,
                compute_device=device.type,
            )
        )
        plan = planner.plan(embedding_module, sharders)
        _append_worker_debug(cfg, rank, "after_sharding_plan")
        embedding_module = DistributedModelParallel(
            module=embedding_module,
            device=device,
            sharders=sharders,
            plan=plan,
        )
        _append_worker_debug(cfg, rank, "after_distributed_model_parallel")
        collective_mode = "measured_distributed"
        collective_measured = 1
    else:
        embedding_module = embedding_module.to(device)
        collective_mode = "not_measured_single_process"
        collective_measured = 0

    dense_input_dim = cfg.embedding_dim * len(DEFAULT_CAT_NAMES) + 13
    dense_module = build_dense_stack(torch, dense_input_dim).to(device)
    if use_dist:
        _append_worker_debug(cfg, rank, "skip_dense_ddp")

    criterion = nn.BCEWithLogitsLoss()
    _append_worker_debug(cfg, rank, "after_criterion")
    _append_worker_debug(cfg, rank, "before_optimizer_init")
    dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)
    sparse_optimizer = torch.optim.SGD(embedding_module.parameters(), lr=0.01)
    _append_worker_debug(cfg, rank, "after_optimizer_init")

    profiler = build_torchrec_profiler(
        cfg,
        on_trace_ready=_make_trace_handler(cfg, rank) if cfg.torchrec_profiler else None,
    )
    profiler_context = profiler or nullcontext()
    _append_worker_debug(cfg, rank, "before_training_loop")

    rows: list[dict[str, Any]] = []
    with profiler_context:
        for step in range(cfg.steps):
            _append_worker_debug(cfg, rank, f"step_start step={step}")
            row: dict[str, Any] = {
                "backend": "torchrec",
                "nproc": world_size,
                "rank": rank,
                "batch_size": cfg.batch_size,
                "step": step,
                "warmup_excluded": _bool_int(step < cfg.warmup_steps),
                "collective_mode": collective_mode,
                "collective_measured": collective_measured,
                "nnodes": cfg.nnodes,
                "nproc_per_node": cfg.nproc_per_node,
                "world_size": cfg.nnodes * cfg.nproc_per_node,
                "dist_mode": "multi_node" if cfg.nnodes > 1 else "single_node",
            }
            step_start = time.perf_counter()

            with stage_timer(row, "batch_prepare_ms"):
                _append_worker_debug(cfg, rank, f"before_batch_prepare step={step}")
                try:
                    dense_batch, sparse_batch, labels_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    dense_batch, sparse_batch, labels_batch = next(data_iter)
                _append_worker_debug(cfg, rank, f"after_batch_prepare step={step}")

            with stage_timer(row, "input_pack_ms"):
                _append_worker_debug(cfg, rank, f"before_input_pack step={step}")
                dense_batch, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                    dense_batch,
                    sparse_batch,
                    labels_batch,
                )
                _append_worker_debug(cfg, rank, f"after_input_pack step={step}")

            if use_dist and device.type == "cuda":
                torch.cuda.synchronize(device)
            collective_start = time.perf_counter()
            with stage_timer(row, "embed_lookup_local_ms"):
                _append_worker_debug(cfg, rank, f"before_embedding step={step}")
                embeddings = embedding_module(sparse_features.to(device))
                _append_worker_debug(cfg, rank, f"after_embedding step={step}")
            if use_dist and device.type == "cuda":
                torch.cuda.synchronize(device)
            collective_elapsed_ms = (time.perf_counter() - collective_start) * 1e3

            with stage_timer(row, "embed_pool_local_ms"):
                _append_worker_debug(cfg, rank, f"before_pool step={step}")
                pooled = torch.cat([embeddings[key] for key in DEFAULT_CAT_NAMES], dim=1)
                pooled = pooled.detach().contiguous()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                _append_worker_debug(cfg, rank, f"after_pool step={step}")

            if use_dist:
                row["collective_launch_ms"] = 0.0
                row["collective_wait_ms"] = collective_elapsed_ms
            else:
                row["collective_launch_ms"] = 0.0
                row["collective_wait_ms"] = 0.0

            with stage_timer(row, "output_unpack_ms"):
                _append_worker_debug(cfg, rank, f"before_output_unpack step={step}")
                dense_input, pooled, labels = prepare_dense_input(
                    dense_batch=dense_batch,
                    pooled_source=pooled,
                    labels_batch=labels_batch,
                    torch=torch,
                    device=device,
                )
                _append_worker_debug(cfg, rank, f"after_output_unpack step={step}")

            with stage_timer(row, "dense_fwd_ms"):
                _append_worker_debug(cfg, rank, f"before_dense_fwd step={step}")
                sync_device(torch, device)
                logits = dense_module(dense_input)
                loss = criterion(logits, labels)
                sync_device(torch, device)
                _append_worker_debug(cfg, rank, f"after_dense_fwd step={step}")

            with stage_timer(row, "backward_ms"):
                _append_worker_debug(cfg, rank, f"before_backward step={step}")
                pooled_grad = run_dense_backward(
                    loss=loss,
                    pooled=pooled,
                    dense_module=dense_module,
                    torch=torch,
                    device=device,
                )
                _append_worker_debug(cfg, rank, f"after_backward step={step}")

            with stage_timer(row, "optimizer_ms"):
                _append_worker_debug(cfg, rank, f"before_optimizer step={step}")
                sync_device(torch, device)
                dense_optimizer.step()
                dense_optimizer.zero_grad(set_to_none=True)
                sync_device(torch, device)
                _append_worker_debug(cfg, rank, f"after_optimizer step={step}")

            with stage_timer(row, "sparse_update_ms"):
                _append_worker_debug(cfg, rank, f"before_sparse_update step={step}")
                sync_device(torch, device)
                pooled.backward(pooled_grad)
                sparse_optimizer.step()
                sparse_optimizer.zero_grad(set_to_none=True)
                sync_device(torch, device)
                _append_worker_debug(cfg, rank, f"after_sparse_update step={step}")

            if profiler is not None:
                profiler.step()

            row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
            rows.append(finalize_torchrec_row(row))

    _append_worker_debug(cfg, rank, f"before_write_rows count={len(rows)} out_csv={out_csv}")
    _write_rows(out_csv, rows)
    _append_worker_debug(cfg, rank, "after_write_rows")
    if is_dist and dist.is_initialized():
        _append_worker_debug(cfg, rank, "before_barrier")
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
        _append_worker_debug(cfg, rank, "after_barrier")
        dist.destroy_process_group()
        _append_worker_debug(cfg, rank, "after_destroy_process_group")
    return rows


class TorchRecRunner(BenchmarkRunner):
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _rank_output_dir(self, cfg: RunConfig) -> Path:
        return Path(cfg.output_root) / "outputs" / cfg.run_id / "torchrec_ranks"

    def _build_torchrun_cmd(self, repo_root: Path, cfg: RunConfig) -> list[str]:
        rdzv_endpoint = f"{cfg.master_addr}:{cfg.master_port}"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(cfg.nnodes),
            "--node_rank",
            str(cfg.node_rank),
            "--nproc_per_node",
            str(cfg.nproc_per_node),
            "--rdzv_backend",
            str(cfg.rdzv_backend),
            "--rdzv_endpoint",
            rdzv_endpoint,
            "--rdzv_id",
            str(cfg.rdzv_id),
            "--tee",
            "3",
            str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
            "--backend",
            "torchrec",
            "--nnodes",
            str(cfg.nnodes),
            "--node-rank",
            str(cfg.node_rank),
            "--nproc-per-node",
            str(cfg.nproc_per_node),
            "--master-addr",
            str(cfg.master_addr),
            "--master-port",
            str(cfg.master_port),
            "--rdzv-backend",
            str(cfg.rdzv_backend),
            "--rdzv-id",
            str(cfg.rdzv_id),
            "--run-id",
            str(cfg.run_id),
            "--output-root",
            str(cfg.output_root),
            "--steps",
            str(cfg.steps),
            "--warmup-steps",
            str(cfg.warmup_steps),
            "--batch-size",
            str(cfg.batch_size),
            "--num-embeddings",
            str(cfg.num_embeddings),
            "--embedding-dim",
            str(cfg.embedding_dim),
            "--seed",
            str(cfg.seed),
            "--data-dir",
            cfg.data_dir,
            "--train-ratio",
            str(cfg.train_ratio),
            "--torchrec-main-csv",
            str(Path(cfg.torchrec_main_csv)),
            "--torchrec-main-agg-csv",
            str(Path(cfg.torchrec_main_agg_csv)),
            "--torchrec-trace-dir",
            str(Path(cfg.torchrec_trace_dir)),
            "--torchrec-trace-csv",
            str(Path(cfg.torchrec_trace_csv)),
            "--no-start-server",
        ]
        if cfg.torchrec_profiler:
            cmd.extend(
                [
                    "--torchrec-profiler",
                    "--torchrec-profiler-warmup",
                    str(cfg.torchrec_profiler_warmup),
                    "--torchrec-profiler-active",
                    str(cfg.torchrec_profiler_active),
                    "--torchrec-profiler-repeat",
                    str(cfg.torchrec_profiler_repeat),
                ]
            )
        return cmd

    def _run_single_process(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rows = _run_single_or_dist_worker(
            repo_root=repo_root,
            cfg=cfg,
            rank=0,
            world_size=1,
            local_rank=0,
            out_csv=Path(cfg.torchrec_main_csv),
        )
        return {"backend": "torchrec", "rows": rows}

    def _run_distributed(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rank_dir = self._rank_output_dir(cfg)
        ensure_shared_dir(rank_dir)

        cmd = self._build_torchrun_cmd(repo_root, cfg)

        env = os.environ.copy()
        env["RS_DEMO_TORCHREC_WORKER"] = "1"
        env["RS_DEMO_TORCHREC_WORKER_DIR"] = str(rank_dir)
        socket_ifname = _pick_socket_ifname()
        if socket_ifname:
            env.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
            env.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
        env.setdefault("NCCL_DEBUG", "WARN")
        res = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                "torchrun worker failed\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}"
            )

        world_size = cfg.nnodes * cfg.nproc_per_node
        rank_csvs = [rank_dir / f"rank{rank}.csv" for rank in range(world_size)]
        missing = [str(path) for path in rank_csvs if not path.exists()]
        if missing:
            raise RuntimeError(f"missing rank csv outputs: {missing}")
        rows = _merge_rank_outputs(rank_csvs, Path(cfg.torchrec_main_csv))
        return {"backend": "torchrec", "rows": rows}

    def run(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        if cfg.backend != "torchrec":
            raise ValueError("TorchRecRunner requires cfg.backend to be 'torchrec'.")
        validate_torchrec_config(cfg)
        if cfg.steps <= 0:
            raise ValueError("TorchRec runner requires --steps to be greater than 0.")

        ensure_torchrec_available()

        if os.environ.get("RS_DEMO_TORCHREC_WORKER") == "1":
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(
                os.environ.get("WORLD_SIZE", str(cfg.nnodes * cfg.nproc_per_node))
            )
            worker_dir = Path(os.environ["RS_DEMO_TORCHREC_WORKER_DIR"])
            ensure_shared_dir(worker_dir)
            out_csv = worker_dir / f"rank{rank}.csv"
            rows = _run_single_or_dist_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                out_csv=out_csv,
            )
            return {"backend": "torchrec", "rows": rows}

        if cfg.nnodes * cfg.nproc_per_node <= 1:
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
