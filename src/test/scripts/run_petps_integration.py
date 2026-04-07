#!/usr/bin/env python3

import argparse
import sys

from petps_cluster_runner import PetPSClusterRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-count", type=int, required=True)
    parser.add_argument("--test-binary", required=True)
    parser.add_argument("--gtest-filter", required=True)
    parser.add_argument("--config-path", default="./recstore_config.json")
    parser.add_argument("--value-size", type=int, default=16)
    parser.add_argument("--max-kv-num-per-request", type=int, default=64)
    parser.add_argument("--client-count", type=int, default=1)
    args = parser.parse_args()

    runner = PetPSClusterRunner(
        config_path=args.config_path,
        num_servers=args.server_count,
        num_clients=args.client_count,
        thread_num=1,
        value_size=args.value_size,
        max_kv_num_per_request=args.max_kv_num_per_request,
    )

    with runner.run():
        completed = runner.run_client(
            [args.test_binary, f"--gtest_filter={args.gtest_filter}"]
        )

    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
