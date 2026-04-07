#!/usr/bin/env python3

import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class PetPSClusterRunner:
    def __init__(
        self,
        server_path="./build/bin/petps_server",
        config_path="./recstore_config.json",
        num_servers=1,
        num_clients=1,
        thread_num=1,
        value_size=16,
        max_kv_num_per_request=64,
        timeout=60,
        startup_delay=2.0,
        log_dir="/tmp",
        verbose=False,
    ):
        self.server_path = Path(server_path)
        if not self.server_path.is_absolute():
            self.server_path = (REPO_ROOT / self.server_path).resolve()

        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (REPO_ROOT / self.config_path).resolve()
        self.num_servers = num_servers
        self.num_clients = num_clients
        self.thread_num = thread_num
        self.value_size = value_size
        self.max_kv_num_per_request = max_kv_num_per_request
        self.timeout = timeout
        self.startup_delay = startup_delay
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.processes = []
        self.ready = set()

    def build_server_cmd(self, global_id):
        return [
            str(self.server_path),
            f"--config_path={self.config_path}",
            f"--global_id={global_id}",
            f"--num_server_processes={self.num_servers}",
            f"--num_client_processes={self.num_clients}",
            f"--thread_num={self.thread_num}",
            f"--value_size={self.value_size}",
            f"--max_kv_num_per_request={self.max_kv_num_per_request}",
        ]

    def build_client_cmd(self, argv, client_index=0):
        client_global_id = self.num_servers + client_index
        return list(argv) + [
            f"--global_id={client_global_id}",
            f"--num_server_processes={self.num_servers}",
            f"--num_client_processes={self.num_clients}",
            f"--value_size={self.value_size}",
            f"--max_kv_num_per_request={self.max_kv_num_per_request}",
        ]

    def is_ready_line(self, line):
        return "Starts PS polling thread" in line or "xmh: finish construct DSM" in line

    def _monitor(self, global_id, pipe):
        for raw_line in iter(pipe.readline, ""):
            line = raw_line.rstrip()
            if self.verbose:
                print(f"[petps_server:{global_id}] {line}")
            if self.is_ready_line(line):
                self.ready.add(global_id)

    def start(self):
        if not self.server_path.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_path}")

        for global_id in range(self.num_servers):
            process = subprocess.Popen(
                self.build_server_cmd(global_id),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(REPO_ROOT),
            )
            thread = threading.Thread(
                target=self._monitor, args=(global_id, process.stdout), daemon=True
            )
            thread.start()
            self.processes.append((process, thread))

        if self.startup_delay > 0:
            time.sleep(self.startup_delay)

        if not self.ready:
            for global_id, (process, _thread) in enumerate(self.processes):
                if process.poll() is None:
                    self.ready.add(global_id)

        deadline = time.time() + self.timeout
        while len(self.ready) < self.num_servers:
            if time.time() > deadline:
                self.stop()
                raise TimeoutError(
                    f"Timed out waiting for {self.num_servers} petps_server processes"
                )
            for process, _thread in self.processes:
                if process.poll() is not None:
                    self.stop()
                    raise RuntimeError(
                        f"petps_server exited early with code {process.returncode}"
                    )
            time.sleep(0.2)

    def run_client(self, argv, client_index=0):
        completed = subprocess.run(
            self.build_client_cmd(argv, client_index=client_index),
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if self.verbose:
            print(completed.stdout)
            print(completed.stderr)
        return completed

    def stop(self):
        for process, thread in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            thread.join(timeout=1)
        self.processes.clear()

    @contextmanager
    def run(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()
