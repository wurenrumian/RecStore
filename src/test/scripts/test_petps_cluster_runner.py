import unittest
from unittest import mock

from petps_cluster_runner import LOCAL_MEMCACHED_SERVER, PetPSClusterRunner


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

    def test_build_env_includes_local_memcached_override(self):
        runner = PetPSClusterRunner(num_servers=2, num_clients=1, memcached_port=12345)
        env = runner.build_env()
        self.assertEqual(env["RECSTORE_MEMCACHED_HOST"], "127.0.0.1")
        self.assertEqual(env["RECSTORE_MEMCACHED_PORT"], "12345")
        self.assertEqual(env["RECSTORE_MEMCACHED_TEXT_PROTOCOL"], "1")

    def test_build_memcached_cmd_uses_local_helper(self):
        runner = PetPSClusterRunner(memcached_port=12345)
        cmd = runner.build_memcached_cmd()
        self.assertEqual(cmd[0], "python3")
        self.assertEqual(cmd[1], str(LOCAL_MEMCACHED_SERVER))
        self.assertIn("--port", cmd)

    @mock.patch("petps_cluster_runner.socket.create_connection")
    def test_memcached_preflight_success(self, mock_conn):
        runner = PetPSClusterRunner(use_local_memcached="never")
        conn = mock.MagicMock()
        conn.recv.return_value = b"END\r\n"
        mock_conn.return_value.__enter__.return_value = conn
        runner.check_memcached_ready()

    @mock.patch("petps_cluster_runner.socket.create_connection")
    def test_memcached_preflight_failure_raises(self, mock_conn):
        runner = PetPSClusterRunner(use_local_memcached="never")
        mock_conn.side_effect = OSError("refused")
        with self.assertRaises(RuntimeError):
            runner.check_memcached_ready()

    @mock.patch("petps_cluster_runner.subprocess.Popen")
    def test_memcached_helper_fallback_to_external(self, mock_popen):
        runner = PetPSClusterRunner(use_local_memcached="auto")
        mock_popen.side_effect = PermissionError("Operation not permitted")
        runner._start_memcached()
        self.assertIsNone(runner.memcached_process)


if __name__ == "__main__":
    unittest.main()
