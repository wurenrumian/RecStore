import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_petps_integration import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    resolve_config_path,
)


class TestRunPetPSIntegration(unittest.TestCase):
    def test_uses_single_shard_rdma_config_by_default(self):
        self.assertEqual(
            resolve_config_path(server_count=1, config_path=None),
            DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        )

    def test_uses_multi_shard_rdma_config_for_multi_server_runs(self):
        self.assertEqual(
            resolve_config_path(server_count=2, config_path=None),
            DEFAULT_RDMA_MULTI_SHARD_CONFIG,
        )

    def test_explicit_config_path_wins(self):
        self.assertEqual(
            resolve_config_path(server_count=2, config_path="./custom.json"),
            "./custom.json",
        )


if __name__ == "__main__":
    unittest.main()
