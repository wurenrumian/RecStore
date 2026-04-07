#!/usr/bin/env python3

import argparse
import subprocess

from petps_cluster_runner import PetPSClusterRunner
from ps_server_runner import PSServerRunner


def run_cmd(cmd):
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    print(completed.stdout, end="")
    print(completed.stderr, end="")
    return completed.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-binary", required=True)
    parser.add_argument("--grpc-config", default="./recstore_config.json")
    parser.add_argument(
        "--brpc-config",
        default="./src/test/scripts/recstore_config.brpc.json",
    )
    args = parser.parse_args()

    rdma_runner = PetPSClusterRunner(num_servers=1, num_clients=1)
    with rdma_runner.run():
        rc = rdma_runner.run_client(
            [args.benchmark_binary, "--transport=rdma", "--num_shards=1", "--iterations=20"]
        )
        print(rc.stdout, end="")
        if rc.returncode != 0:
            return rc.returncode

    grpc_runner = PSServerRunner(config_path=args.grpc_config, num_shards=2)
    with grpc_runner.run():
        rc = run_cmd(
            [
                args.benchmark_binary,
                "--transport=grpc",
                "--host=127.0.0.1",
                "--port=25000",
                "--iterations=20",
            ]
        )
        if rc != 0:
            return rc

    brpc_runner = PSServerRunner(config_path=args.brpc_config, num_shards=2)
    with brpc_runner.run():
        rc = run_cmd(
            [
                args.benchmark_binary,
                "--transport=brpc",
                "--host=127.0.0.1",
                "--port=25000",
                "--iterations=20",
            ]
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
