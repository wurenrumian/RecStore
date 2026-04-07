import unittest

from petps_cluster_runner import PetPSClusterRunner


class TestPetPSClusterRunner(unittest.TestCase):
    def test_build_server_command_includes_postoffice_flags(self):
        runner = PetPSClusterRunner(
            server_path="./build/bin/petps_server",
            config_path="./recstore_config.json",
            num_servers=2,
            num_clients=1,
            thread_num=2,
            value_size=16,
            max_kv_num_per_request=64,
        )

        cmd = runner.build_server_cmd(1)

        self.assertIn("--global_id=1", cmd)
        self.assertIn("--num_server_processes=2", cmd)
        self.assertIn("--num_client_processes=1", cmd)
        self.assertIn("--thread_num=2", cmd)
        self.assertIn("--value_size=16", cmd)
        self.assertIn("--max_kv_num_per_request=64", cmd)

    def test_detects_ready_lines(self):
        runner = PetPSClusterRunner()
        self.assertTrue(runner.is_ready_line("Starts PS polling thread 0"))
        self.assertTrue(runner.is_ready_line("xmh: finish construct DSM"))
        self.assertFalse(runner.is_ready_line("throughput 0.1234 Mkv/s"))

    def test_build_client_command_assigns_client_global_id(self):
        runner = PetPSClusterRunner(num_servers=2, num_clients=1)
        cmd = runner.build_client_cmd(["./build/bin/petps_integration_test"], client_index=0)
        self.assertIn("--global_id=2", cmd)
        self.assertIn("--num_server_processes=2", cmd)
        self.assertIn("--num_client_processes=1", cmd)


if __name__ == "__main__":
    unittest.main()
