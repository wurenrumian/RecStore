#!/usr/bin/env python3

import argparse
import subprocess

from petps_cluster_runner import PetPSClusterRunner
from ps_test_config import (
    DEFAULT_BRPC_BENCHMARK_CONFIG,
    DEFAULT_GRPC_MAIN_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    load_client_endpoint,
)
from ps_server_runner import PSServerRunner


MEMCACHED_NOISE_PATTERNS = (
    "[petps-memcached]",
    "[petps-status] phase=memcached",
    "[memcached-endpoint]",
    "use memcached in ",
)


def build_rdma_runner(args):
    return PetPSClusterRunner(
        config_path=DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        num_servers=1,
        num_clients=1,
        use_local_memcached=args.use_local_memcached,
        memcached_host=args.memcached_host,
        memcached_port=args.memcached_port,
        show_status_logs=args.show_runner_logs,
        show_memcached_logs=args.show_runner_logs,
        rdma_per_thread_response_limit_bytes=args.rdma_per_thread_response_limit_bytes,
        rdma_server_ready_timeout_sec=args.rdma_server_ready_timeout_sec,
        rdma_server_ready_poll_ms=args.rdma_server_ready_poll_ms,
        rdma_client_receive_arena_bytes=args.rdma_client_receive_arena_bytes,
        validate_routing=args.validate_routing,
    )

def is_memcached_noise_line(line):
    return any(pattern in line for pattern in MEMCACHED_NOISE_PATTERNS)


def print_filtered_output(text, show_runner_logs):
    for line in text.splitlines():
        if not show_runner_logs and is_memcached_noise_line(line):
            continue
        print(line)


def run_cmd(cmd, show_runner_logs):
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    print_filtered_output(completed.stdout, show_runner_logs)
    print_filtered_output(completed.stderr, show_runner_logs)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-binary", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rdma-warmup-rounds", type=int, default=2)
    parser.add_argument("--grpc-config", default=DEFAULT_GRPC_MAIN_CONFIG)
    parser.add_argument(
        "--brpc-config",
        default=DEFAULT_BRPC_BENCHMARK_CONFIG,
    )
    parser.add_argument(
        "--use-local-memcached",
        choices=["always", "auto", "never"],
        default="auto",
    )
    parser.add_argument("--memcached-host", default="127.0.0.1")
    parser.add_argument("--memcached-port", type=int, default=21211)
    parser.add_argument("--rdma-per-thread-response-limit-bytes", type=int)
    parser.add_argument("--rdma-server-ready-timeout-sec", type=int)
    parser.add_argument("--rdma-server-ready-poll-ms", type=int)
    parser.add_argument("--rdma-client-receive-arena-bytes", type=int)
    parser.add_argument("--validate-routing", action="store_true")
    parser.add_argument(
        "--show-runner-logs",
        action="store_true",
        help="show memcached/status logs from runner and benchmark binaries",
    )
    args = parser.parse_args()

    print("[阶段] 1/3 RDMA benchmark")
    rdma_runner = build_rdma_runner(args)
    with rdma_runner.run():
        completed = rdma_runner.run_client(
            [
                args.benchmark_binary,
                "--transport=rdma",
                "--num_shards=1",
                f"--iterations={args.iterations}",
                f"--rounds={args.rounds}",
                f"--warmup_rounds={args.rdma_warmup_rounds}",
            ],
            stream_output=False,
        )
        print_filtered_output(completed.stdout, args.show_runner_logs)
        print_filtered_output(completed.stderr, args.show_runner_logs)
        rc = completed.returncode
        if rc != 0:
            return rc

    print("[阶段] 2/3 GRPC benchmark")
    grpc_host, grpc_port = load_client_endpoint(args.grpc_config)
    grpc_runner = PSServerRunner(config_path=args.grpc_config, num_shards=2)
    with grpc_runner.run():
        rc = run_cmd(
            [
                args.benchmark_binary,
                "--transport=grpc",
                f"--host={grpc_host}",
                f"--port={grpc_port}",
                f"--iterations={args.iterations}",
                f"--rounds={args.rounds}",
                "--warmup_rounds=0",
            ],
            args.show_runner_logs,
        )
        if rc != 0:
            return rc

    print("[阶段] 3/3 BRPC benchmark")
    brpc_host, brpc_port = load_client_endpoint(args.brpc_config)
    brpc_runner = PSServerRunner(config_path=args.brpc_config, num_shards=2)
    with brpc_runner.run():
        rc = run_cmd(
            [
                args.benchmark_binary,
                "--transport=brpc",
                f"--host={brpc_host}",
                f"--port={brpc_port}",
                f"--iterations={args.iterations}",
                f"--rounds={args.rounds}",
                "--warmup_rounds=0",
            ],
            args.show_runner_logs,
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
