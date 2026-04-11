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


def build_rdma_runner(args):
    return PetPSClusterRunner(
        config_path=DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        num_servers=1,
        num_clients=1,
        use_local_memcached=args.use_local_memcached,
        memcached_host=args.memcached_host,
        memcached_port=args.memcached_port,
    )

def run_cmd(cmd):
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    print(completed.stdout, end="")
    print(completed.stderr, end="")
    return completed.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-binary", required=True)
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
    args = parser.parse_args()

    rdma_runner = build_rdma_runner(args)
    with rdma_runner.run():
        rc = rdma_runner.run_client(
            [args.benchmark_binary, "--transport=rdma", "--num_shards=1", "--iterations=20"]
        )
        # Output is already streamed in run_client, no need to print again
        if rc.returncode != 0:
            return rc.returncode

    grpc_host, grpc_port = load_client_endpoint(args.grpc_config)
    grpc_runner = PSServerRunner(config_path=args.grpc_config, num_shards=2)
    with grpc_runner.run():
        rc = run_cmd(
            [
                args.benchmark_binary,
                "--transport=grpc",
                f"--host={grpc_host}",
                f"--port={grpc_port}",
                "--iterations=20",
            ]
        )
        if rc != 0:
            return rc

    brpc_host, brpc_port = load_client_endpoint(args.brpc_config)
    brpc_runner = PSServerRunner(config_path=args.brpc_config, num_shards=2)
    with brpc_runner.run():
        rc = run_cmd(
            [
                args.benchmark_binary,
                "--transport=brpc",
                f"--host={brpc_host}",
                f"--port={brpc_port}",
                "--iterations=20",
            ]
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
