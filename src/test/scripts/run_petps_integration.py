#!/usr/bin/env python3

import argparse

from petps_cluster_runner import PetPSClusterRunner
from ps_test_config import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    resolve_rdma_integration_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-count", type=int, required=True)
    parser.add_argument("--test-binary", required=True)
    parser.add_argument("--gtest-filter", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--value-size", type=int, default=16)
    parser.add_argument("--max-kv-num-per-request", type=int, default=64)
    parser.add_argument("--client-count", type=int, default=1)
    parser.add_argument("--client-timeout", type=int, default=120)
    parser.add_argument(
        "--use-local-memcached",
        choices=["always", "auto", "never"],
        default="auto",
    )
    parser.add_argument("--memcached-host", default="127.0.0.1")
    parser.add_argument("--memcached-port", type=int, default=21211)
    args = parser.parse_args()
    config_path = resolve_rdma_integration_config(args.server_count, args.config_path)

    runner = PetPSClusterRunner(
        config_path=config_path,
        num_servers=args.server_count,
        num_clients=args.client_count,
        thread_num=1,
        value_size=args.value_size,
        max_kv_num_per_request=args.max_kv_num_per_request,
        use_local_memcached=args.use_local_memcached,
        memcached_host=args.memcached_host,
        memcached_port=args.memcached_port,
    )

    with runner.run():
        completed = runner.run_client(
            [args.test_binary, f"--gtest_filter={args.gtest_filter}"],
            timeout=args.client_timeout,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
