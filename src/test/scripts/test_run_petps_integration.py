import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps_test_config import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    resolve_rdma_integration_config,
)
from run_petps_integration import MAX_TIMEOUT_SECONDS, normalize_timeout


class TestRunPetPSIntegration(unittest.TestCase):
    def test_timeout_is_capped_to_hard_limit(self):
        self.assertEqual(normalize_timeout(99), MAX_TIMEOUT_SECONDS)

    def test_timeout_keeps_value_within_limit(self):
        self.assertEqual(normalize_timeout(10), 10)

    def test_uses_single_shard_rdma_config_by_default(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=1, config_path=None),
            DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        )

    def test_uses_multi_shard_rdma_config_for_multi_server_runs(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path=None),
            DEFAULT_RDMA_MULTI_SHARD_CONFIG,
        )

    def test_explicit_config_path_wins(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path="./custom.json"),
            "./custom.json",
        )


if __name__ == "__main__":
    unittest.main()
