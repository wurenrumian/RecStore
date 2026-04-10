#!/usr/bin/env python3

import os
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MEMCACHED_SERVER = REPO_ROOT / "src/test/scripts/local_memcached_server.py"


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
        memcached_host="127.0.0.1",
        memcached_port=21211,
        use_local_memcached="auto",
        memcached_check_timeout=2.0,
        memcached_check_retries=3,
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
        self.memcached_host = memcached_host
        self.memcached_port = memcached_port
        self.use_local_memcached = use_local_memcached
        self.memcached_check_timeout = memcached_check_timeout
        self.memcached_check_retries = memcached_check_retries
        self.processes = []
        self.memcached_process = None
        self.ready = set()

    def build_env(self):
        env = os.environ.copy()
        env["RECSTORE_MEMCACHED_HOST"] = self.memcached_host
        env["RECSTORE_MEMCACHED_PORT"] = str(self.memcached_port)
        env["RECSTORE_MEMCACHED_TEXT_PROTOCOL"] = "1"
        return env

    def build_memcached_cmd(self):
        return [
            "python3",
            str(LOCAL_MEMCACHED_SERVER),
            "--host",
            self.memcached_host,
            "--port",
            str(self.memcached_port),
        ]

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

    def _start_memcached(self):
        try:
            self.memcached_process = subprocess.Popen(
                self.build_memcached_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(REPO_ROOT),
            )
        except Exception as exc:
            if self.use_local_memcached == "always":
                raise RuntimeError(f"failed to launch local memcached helper: {exc}") from exc
            self.memcached_process = None
            return

        time.sleep(0.2)
        if self.memcached_process.poll() is not None:
            log = ""
            if self.memcached_process.stdout is not None:
                log = self.memcached_process.stdout.read()
            err = (
                f"local memcached helper exited early with code {self.memcached_process.returncode}"
            )
            if log:
                err += f"\nhelper output:\n{log}"
            if self.use_local_memcached == "always":
                raise RuntimeError(err)
            self.memcached_process = None

    def check_memcached_ready(self):
        last_error = None
        for _ in range(self.memcached_check_retries):
            try:
                with socket.create_connection(
                    (self.memcached_host, self.memcached_port),
                    timeout=self.memcached_check_timeout,
                ) as sock:
                    sock.settimeout(self.memcached_check_timeout)
                    sock.sendall(b"get serverNum\r\n")
                    data = sock.recv(4096)
                    if b"END\r\n" in data or b"VALUE serverNum" in data:
                        return
                    last_error = RuntimeError(
                        "memcached responded but without expected get reply"
                    )
            except OSError as exc:
                last_error = exc
            time.sleep(0.2)

        raise RuntimeError(
            "memcached is not reachable or not ready at "
            f"{self.memcached_host}:{self.memcached_port}; "
            "set --use-local-memcached=always|auto|never and "
            "RECSTORE_MEMCACHED_HOST/RECSTORE_MEMCACHED_PORT as needed"
        ) from last_error

    def reset_memcached_state(self):
        payload = (
            b"set serverNum 0 0 1\r\n0\r\n"
            b"set clientNum 0 0 1\r\n0\r\n"
            b"set xmh-consistent-dsm 0 0 1\r\n1\r\n"
            b"quit\r\n"
        )
        with socket.create_connection(
            (self.memcached_host, self.memcached_port),
            timeout=self.memcached_check_timeout,
        ) as sock:
            sock.settimeout(self.memcached_check_timeout)
            sock.sendall(payload)
            response = b""
            while True:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                response += chunk
            if response.count(b"STORED") < 3:
                raise RuntimeError(
                    "failed to initialize memcached state; "
                    f"response was: {response!r}"
                )

    def start(self):
        if not self.server_path.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_path}")

        if self.use_local_memcached not in {"always", "auto", "never"}:
            raise ValueError(
                "use_local_memcached must be one of: always, auto, never"
            )
        if self.use_local_memcached in {"always", "auto"}:
            self._start_memcached()
        self.check_memcached_ready()
        self.reset_memcached_state()
        env = self.build_env()

        for global_id in range(self.num_servers):
            process = subprocess.Popen(
                self.build_server_cmd(global_id),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(REPO_ROOT),
                env=env,
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

    def run_client(self, argv, client_index=0, stream_output=True, timeout=None):
        cmd = self.build_client_cmd(argv, client_index=client_index)
        if not stream_output:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                check=False,
                env=self.build_env(),
                timeout=timeout,
            )
            if self.verbose:
                print(completed.stdout)
                print(completed.stderr)
            return completed

        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=self.build_env(),
        )

        output_lines = []
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                output_lines.append(line)
                print(line, end="")
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            timeout_line = f"[petps-client] timed out after {timeout} seconds\n"
            output_lines.append(timeout_line)
            print(timeout_line, end="")
            returncode = 124

        class Completed:
            def __init__(self, returncode, stdout):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = ""

        return Completed(returncode, "".join(output_lines))

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
        if self.memcached_process is not None and self.memcached_process.poll() is None:
            self.memcached_process.terminate()
            try:
                self.memcached_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.memcached_process.kill()
                self.memcached_process.wait(timeout=5)
        self.memcached_process = None

    @contextmanager
    def run(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()
