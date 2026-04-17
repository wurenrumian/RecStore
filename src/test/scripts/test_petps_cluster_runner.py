import unittest
from unittest import mock
from io import StringIO

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
        self.assertTrue(
            runner.is_ready_line("[RDMA-DBG] Server polling thread ready 0")
        )
        self.assertFalse(runner.is_ready_line("Starts PS polling thread 0"))
        self.assertFalse(runner.is_ready_line("xmh: finish construct DSM"))
        self.assertFalse(runner.is_ready_line("throughput 0.1234 Mkv/s"))

    def test_monitor_requires_all_polling_threads_before_marking_ready(self):
        runner = PetPSClusterRunner(num_servers=1, thread_num=2)
        pipe = StringIO(
            "xmh: finish construct DSM\n"
            "[RDMA-DBG] Server polling thread ready 0\n"
            "throughput 0.1234 Mkv/s\n"
            "[RDMA-DBG] Server polling thread ready 1\n"
        )

        runner._monitor(0, pipe)

        self.assertEqual(runner.ready, {0})

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

    @mock.patch("petps_cluster_runner.shutil.which", return_value="/usr/bin/memcached")
    def test_build_memcached_cmd_uses_system_memcached(self, _mock_which):
        runner = PetPSClusterRunner(memcached_port=12345)
        cmd = runner.build_memcached_cmd()
        self.assertEqual(cmd[0], "/usr/bin/memcached")
        self.assertEqual(cmd[1:5], ["-u", "root", "-l", "127.0.0.1"])
        self.assertIn("12345", cmd)

    @mock.patch("petps_cluster_runner.shutil.which", return_value=None)
    def test_build_memcached_cmd_requires_system_binary(self, _mock_which):
        runner = PetPSClusterRunner(memcached_port=12345)
        with self.assertRaises(RuntimeError):
            runner.build_memcached_cmd()

    @mock.patch("petps_cluster_runner.socket.create_connection")
    def test_reset_memcached_state_flushes_and_verifies_keys(self, mock_conn):
        runner = PetPSClusterRunner(use_local_memcached="never")
        conn = mock.MagicMock()
        conn.recv.side_effect = [
            (
                b"OK\r\n"
                b"STORED\r\nSTORED\r\nSTORED\r\n"
                b"VALUE serverNum 0 1\r\n0\r\nEND\r\n"
                b"VALUE clientNum 0 1\r\n0\r\nEND\r\n"
                b"VALUE xmh-consistent-dsm 0 1\r\n1\r\nEND\r\n"
            ),
            b"",
        ]
        mock_conn.return_value.__enter__.return_value = conn

        runner.reset_memcached_state()

        sent = b"".join(call.args[0] for call in conn.sendall.call_args_list)
        self.assertIn(b"flush_all\r\n", sent)
        self.assertIn(b"get serverNum\r\n", sent)
        self.assertIn(b"get clientNum\r\n", sent)
        self.assertIn(b"get xmh-consistent-dsm\r\n", sent)

    def test_auto_memcached_falls_back_to_system_memcached_when_reset_fails(self):
        runner = PetPSClusterRunner(use_local_memcached="auto")
        runner.server_path = mock.Mock()
        runner.server_path.exists.return_value = True
        runner.startup_delay = 0
        runner.timeout = 0
        runner.status_refresh_interval = 0

        fake_proc = mock.Mock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 4321
        fake_thread = mock.Mock()

        with mock.patch.object(runner, "_start_memcached") as mock_start_memcached, \
             mock.patch.object(runner, "check_memcached_ready"), \
             mock.patch.object(
                 runner,
                 "reset_memcached_state",
                 side_effect=[RuntimeError("bad reset"), None],
             ) as mock_reset, \
             mock.patch.object(
                 runner, "_allocate_local_memcached_port", return_value=31337
             ), \
             mock.patch("petps_cluster_runner.subprocess.Popen", return_value=fake_proc), \
             mock.patch("petps_cluster_runner.threading.Thread", return_value=fake_thread), \
             mock.patch.object(runner, "stop"):

            def start_memcached():
                runner.memcached_process = mock.Mock()
                runner.memcached_process.poll.return_value = None

            mock_start_memcached.side_effect = start_memcached
            runner.start()

        self.assertEqual(mock_reset.call_count, 2)
        self.assertEqual(runner.memcached_port, 31337)
        self.assertEqual(mock_start_memcached.call_count, 1)

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

    def test_emit_status_prints_ready_and_pid_info(self):
        runner = PetPSClusterRunner(num_servers=2)
        runner.ready.add(0)
        fake_proc = mock.Mock()
        fake_proc.pid = 1234
        fake_proc.poll.return_value = None
        runner.processes = [(fake_proc, mock.Mock())]

        with mock.patch("sys.stdout", new_callable=StringIO) as fake_out:
            runner.emit_status("startup-wait")

        output = fake_out.getvalue()
        self.assertIn("[petps-status]", output)
        self.assertIn("ready=1/2", output)
        self.assertIn("running_pids=1234", output)


if __name__ == "__main__":
    unittest.main()
