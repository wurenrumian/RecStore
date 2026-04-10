import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps_test_config import (
    DEFAULT_BRPC_BENCHMARK_CONFIG,
    DEFAULT_GRPC_MAIN_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
)
from run_rdma_transport_benchmarks import build_rdma_runner, load_client_endpoint


class TestRunRDMATransportBenchmarks(unittest.TestCase):
    def test_rdma_runner_uses_rdma_specific_config(self):
        args = SimpleNamespace(
            use_local_memcached="never",
            memcached_host="127.0.0.1",
            memcached_port=21211,
        )

        runner = build_rdma_runner(args)

        expected = (Path("/app/RecStore") / DEFAULT_RDMA_SINGLE_SHARD_CONFIG).resolve()
        self.assertEqual(runner.config_path, expected)

    def test_load_client_endpoint_for_default_grpc_config(self):
        host, port = load_client_endpoint(DEFAULT_GRPC_MAIN_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 15000)

    def test_load_client_endpoint_for_brpc_config(self):
        host, port = load_client_endpoint(DEFAULT_BRPC_BENCHMARK_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 25000)


if __name__ == "__main__":
    unittest.main()
