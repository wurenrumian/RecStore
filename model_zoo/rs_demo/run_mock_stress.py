#!/usr/bin/env python3
import argparse
import copy
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


def inject_recstore_project_paths(repo_root: Path) -> None:
    recstore_src = str(repo_root / "src")
    dlrm_root = str(repo_root / "model_zoo/torchrec_dlrm")
    py_client = str(repo_root / "src/framework/pytorch/python_client")
    if recstore_src not in sys.path:
        sys.path.insert(0, recstore_src)
    if dlrm_root not in sys.path:
        sys.path.insert(0, dlrm_root)
    if py_client not in sys.path:
        sys.path.insert(0, py_client)


def build_kjt_batch_from_dense_sparse_labels(
    dense_batch: torch.Tensor,
    sparse_batch: torch.Tensor,
    labels_batch: torch.Tensor,
):
    del labels_batch
    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    sparse_mat = sparse_batch.to(torch.long)
    batch_size = sparse_mat.shape[0]
    values_list = [sparse_mat[:, i] for i in range(26)]
    values = torch.cat(values_list, dim=0)
    one_lengths = torch.ones(batch_size, dtype=torch.int32, device=values.device)
    lengths = torch.cat([one_lengths for _ in range(26)], dim=0)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=DEFAULT_CAT_NAMES,
        values=values,
        lengths=lengths,
    )
    return dense_batch, kjt


def build_batch_ids_from_kjt(sparse_features) -> torch.Tensor:
    # Keep exactly the same id source as DLRM: concatenate per-feature values in key order.
    ids_chunks = []
    for key in sparse_features.keys():
        ids_chunks.append(sparse_features[key].values().to(torch.int64))
    if not ids_chunks:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(ids_chunks, dim=0).cpu().contiguous()


def build_table_offsets_from_eb_configs(eb_configs: list[dict], fusion_k: int) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for idx, cfg in enumerate(eb_configs):
        offsets[cfg["feature_names"][0]] = idx << fusion_k
    return offsets


def convert_kjt_ids_to_fused_ids(
    sparse_features,
    table_offsets: dict[str, int],
) -> torch.Tensor:
    ids_chunks = []
    for key in sparse_features.keys():
        vals = sparse_features[key].values().to(torch.int64).cpu()
        ids_chunks.append(vals + table_offsets[key])
    if not ids_chunks:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(ids_chunks, dim=0).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock RecStore stress workload for emb read/update stage profiling."
    )
    parser.add_argument("--num-embeddings", type=int, default=200000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--table-name", type=str, default="mock_perf_table")
    parser.add_argument("--init-rows", type=int, default=50000)
    parser.add_argument("--read-before-update", action="store_true", default=True)
    parser.add_argument("--no-read-before-update", action="store_true")
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port0", type=int, default=None)
    parser.add_argument("--server-port1", type=int, default=None)
    parser.add_argument("--server-wait-seconds", type=float, default=20.0)
    parser.add_argument("--allocator", type=str, default="R2ShmMalloc")
    parser.add_argument("--jsonl", type=str, default="/tmp/recstore_mock_events.jsonl")
    parser.add_argument("--csv", type=str, default="/tmp/recstore_mock_embupdate.csv")
    parser.add_argument(
        "--library-path",
        type=str,
        default="",
        help="Path to RecStore torch ops shared library. Auto-detected if empty.",
    )
    parser.add_argument(
        "--server-log",
        type=str,
        default="/tmp/recstore_mock_ps_server.log",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="model_zoo/torchrec_dlrm/processed_day_0_data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--fuse-k",
        type=int,
        default=30,
        help="Use same fusion prefix rule as DLRM RecStoreEmbeddingBagCollection.",
    )
    return parser.parse_args()


def wait_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sock = socket.socket()
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            time.sleep(0.2)
        finally:
            sock.close()
    return False


def summarize_us(values: list[float]) -> str:
    if not values:
        return "count=0"
    s = sorted(values)
    p50 = s[len(s) // 2]
    p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
    return (
        f"count={len(values)} mean={statistics.fmean(values):.2f}us "
        f"p50={p50:.2f}us p95={p95:.2f}us max={s[-1]:.2f}us"
    )


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


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


def build_runtime_config(
    base_cfg: dict,
    host: str,
    port0: int,
    port1: int,
    allocator: str,
    path_suffix: str,
) -> dict:
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("cache_ps", {})
    cfg["cache_ps"]["ps_type"] = "BRPC"

    cfg.setdefault("client", {})
    cfg["client"]["host"] = host
    cfg["client"]["port"] = port0
    cfg["client"]["shard"] = 0

    cfg.setdefault("distributed_client", {})
    cfg["distributed_client"]["num_shards"] = 2
    cfg["distributed_client"]["servers"] = [
        {"host": host, "port": port0, "shard": 0},
        {"host": host, "port": port1, "shard": 1},
    ]

    cfg["cache_ps"]["num_shards"] = 2
    cfg["cache_ps"]["servers"] = [
        {"host": host, "port": port0, "shard": 0},
        {"host": host, "port": port1, "shard": 1},
    ]

    base_kv = cfg["cache_ps"].setdefault("base_kv_config", {})
    base_kv["value_memory_management"] = allocator
    base_kv["path"] = f"/tmp/recstore_mock_data_{path_suffix}"
    base_kv["index_type"] = "DRAM"
    return cfg


def start_server(repo_root: Path, cfg_path: Path, log_path: Path) -> subprocess.Popen:
    server_bin = repo_root / "build/bin/ps_server"
    if not server_bin.exists():
        raise FileNotFoundError(f"ps_server not found: {server_bin}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(server_bin), "--config_path", str(cfg_path)],
        cwd=str(repo_root),
        stdout=fout,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    proc._mock_log_file = fout  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    log_f = getattr(proc, "_mock_log_file", None)
    if log_f is not None:
        log_f.close()


def main() -> int:
    args = parse_args()
    if args.no_read_before_update:
        args.read_before_update = False
    if args.no_start_server:
        args.start_server = False

    repo_root = repo_root_from_this_file()
    library_path = detect_library_path(repo_root, args.library_path)
    print(f"[mock] repo_root={repo_root}")
    print(f"[mock] library={library_path}")

    with open(repo_root / "recstore_config.json", "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    # Prefer ports from repository config to keep client/server aligned by default.
    distributed_servers = (
        base_cfg.get("distributed_client", {}).get("servers", [])
        or base_cfg.get("cache_ps", {}).get("servers", [])
    )
    default_port0 = None
    default_port1 = None
    if len(distributed_servers) >= 2:
        default_port0 = int(distributed_servers[0]["port"])
        default_port1 = int(distributed_servers[1]["port"])
    elif "client" in base_cfg and "port" in base_cfg["client"]:
        default_port0 = int(base_cfg["client"]["port"])
        default_port1 = default_port0 + 1
    else:
        default_port0 = 15000
        default_port1 = 15001

    if args.server_port0 is None:
        args.server_port0 = default_port0
    if args.server_port1 is None:
        args.server_port1 = default_port1

    now_tag = str(int(time.time()))
    runtime_cfg = build_runtime_config(
        base_cfg=base_cfg,
        host=args.server_host,
        port0=args.server_port0,
        port1=args.server_port1,
        allocator=args.allocator,
        path_suffix=now_tag,
    )
    runtime_dir = Path(f"/tmp/recstore_mock_runtime_{now_tag}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg_path = runtime_dir / "recstore_config.json"
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        json.dump(runtime_cfg, f, indent=2)

    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.server_log).parent.mkdir(parents=True, exist_ok=True)
    open(args.jsonl, "w", encoding="utf-8").close()

    os.environ["RECSTORE_REPORT_MODE"] = "local"
    os.environ["RECSTORE_REPORT_LOCAL_SINK"] = "both"
    os.environ["RECSTORE_REPORT_JSONL_PATH"] = args.jsonl

    proc = None
    orig_cwd = Path.cwd()
    try:
        if args.start_server:
            print(f"[mock] starting server with {runtime_cfg_path}")
            proc = start_server(repo_root, runtime_cfg_path, Path(args.server_log))
            ok0 = wait_port(args.server_host, args.server_port0, args.server_wait_seconds)
            ok1 = wait_port(args.server_host, args.server_port1, args.server_wait_seconds)
            if not (ok0 and ok1):
                raise RuntimeError(
                    f"server ports not ready: {args.server_host}:{args.server_port0},{args.server_port1}"
                )
            print("[mock] server is ready")

        inject_recstore_project_paths(repo_root)
        from data.custom_dataloader import CustomCriteoDataset  # pylint: disable=import-error
        from torchrec.datasets.criteo import DEFAULT_CAT_NAMES  # pylint: disable=import-error
        from client import RecstoreClient  # pylint: disable=import-error

        # Recstore client resolves recstore_config.json from cwd/parent paths.
        # Switch into a runtime directory that contains the generated config.
        os.chdir(str(runtime_dir))

        torch.manual_seed(args.seed)
        client = RecstoreClient(library_path=str(library_path))

        data_dir = (repo_root / args.data_dir).resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"dataset dir not found: {data_dir}")

        nep = [int(args.num_embeddings)] * 26
        dataset = CustomCriteoDataset(
            data_dir=str(data_dir),
            stage="train",
            train_ratio=args.train_ratio,
            num_embeddings_per_feature=nep,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
        )

        # Keep table layout compatible with DLRM RecStore path: one table per sparse feature.
        eb_configs = [
            {
                "name": f"t_{feature_name}",
                "num_embeddings": int(args.num_embeddings),
                "embedding_dim": int(args.embedding_dim),
                "feature_names": [feature_name],
            }
            for feature_name in DEFAULT_CAT_NAMES
        ]

        t0 = time.perf_counter()
        for cfg in eb_configs:
            ok = client.init_embedding_table(
                cfg["name"], cfg["num_embeddings"], cfg["embedding_dim"]
            )
            if not ok:
                raise RuntimeError(f"init_embedding_table failed: {cfg['name']}")
        print(f"[mock] init {len(eb_configs)} tables done in {(time.perf_counter() - t0):.3f}s")

        table_offsets = build_table_offsets_from_eb_configs(eb_configs, args.fuse_k)
        init_rows = min(int(args.init_rows), len(dataset))
        init_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(args.batch_size, init_rows),
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
        )
        init_written = 0
        t0 = time.perf_counter()
        for dense_batch, sparse_batch, labels_batch in init_loader:
            _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                dense_batch, sparse_batch, labels_batch
            )
            fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
            if fused_ids.numel() == 0:
                continue
            vals = torch.randn(fused_ids.numel(), args.embedding_dim, dtype=torch.float32) * 0.01
            client.emb_write(fused_ids, vals)
            init_written += fused_ids.numel()
            if init_written >= init_rows * 26:
                break
        print(
            f"[mock] initial emb_write fused_rows={init_written} in {(time.perf_counter() - t0):.3f}s"
        )

        read_lat_us: list[float] = []
        update_lat_us: list[float] = []
        data_iter = iter(dataloader)
        for step in range(args.steps):
            try:
                dense_batch, sparse_batch, labels_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                dense_batch, sparse_batch, labels_batch = next(data_iter)

            _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                dense_batch, sparse_batch, labels_batch
            )
            # DLRM-like ids for pull path; update uses fused ids (same as torchrec_kv fusion mode).
            read_ids = build_batch_ids_from_kjt(sparse_features)
            fused_ids = convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)
            if fused_ids.numel() == 0:
                continue
            grads = torch.randn(fused_ids.numel(), args.embedding_dim, dtype=torch.float32) * 0.001

            if args.read_before_update:
                t1 = time.perf_counter()
                _ = client.emb_read(read_ids, args.embedding_dim)
                t2 = time.perf_counter()
                if step >= args.warmup_steps:
                    read_lat_us.append((t2 - t1) * 1e6)

            t1 = time.perf_counter()
            client.emb_update_table("t_cat_0", fused_ids, grads)
            t2 = time.perf_counter()
            if step >= args.warmup_steps:
                update_lat_us.append((t2 - t1) * 1e6)

            if (step + 1) % 10 == 0:
                print(
                    f"[mock] step {step + 1}/{args.steps} "
                    f"read={read_lat_us[-1] if read_lat_us else 0:.1f}us "
                    f"update={update_lat_us[-1] if update_lat_us else 0:.1f}us"
                )

        print("[mock] workload finished")
        print(f"[mock] emb_read latency: {summarize_us(read_lat_us)}")
        print(f"[mock] emb_update latency: {summarize_us(update_lat_us)}")

        analyze_cmd = [
            sys.executable,
            str(repo_root / "src/test/scripts/analyze_embupdate_stages.py"),
            "--input",
            args.jsonl,
            "--group-by-prefix",
            "--export-csv",
            args.csv,
            "--top",
            "20",
        ]
        print(f"[mock] analyze: {' '.join(analyze_cmd)}")
        result = subprocess.run(
            analyze_cmd,
            cwd=str(repo_root),
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"analyze script failed with code {result.returncode}")

        print(f"[mock] jsonl: {args.jsonl}")
        print(f"[mock] csv:   {args.csv}")
        print(f"[mock] server log: {args.server_log}")
        return 0
    finally:
        os.chdir(str(orig_cwd))
        stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
