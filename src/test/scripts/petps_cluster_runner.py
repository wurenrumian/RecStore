#!/usr/bin/env python3

import os
import shutil
import socket
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
        memcached_host="127.0.0.1",
        memcached_port=21211,
        use_local_memcached="auto",
        memcached_check_timeout=2.0,
        memcached_check_retries=3,
        status_refresh_interval=2.0,
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
        self.status_refresh_interval = status_refresh_interval
        self.processes = []
        self.memcached_process = None
        self.ready = set()
        self.ready_threads = {}

    def _allocate_local_memcached_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.memcached_host, 0))
            return sock.getsockname()[1]

    def emit_status(self, phase, extra=""):
        running_pids = [
            str(process.pid)
            for process, _thread in self.processes
            if process.poll() is None
        ]
        detail = (
            f" phase={phase} ready={len(self.ready)}/{self.num_servers}"
            f" running_pids={','.join(running_pids) if running_pids else 'none'}"
        )
        if extra:
            detail += f" {extra}"
        print(f"[petps-status]{detail}")

    def build_env(self):
        env = os.environ.copy()
        env["RECSTORE_MEMCACHED_HOST"] = self.memcached_host
        env["RECSTORE_MEMCACHED_PORT"] = str(self.memcached_port)
        env["RECSTORE_MEMCACHED_TEXT_PROTOCOL"] = "1"
        return env

    def build_memcached_cmd(self):
        memcached_bin = shutil.which("memcached")
        if memcached_bin is None:
            raise RuntimeError(
                "memcached binary not found in PATH; install memcached or use "
                "--use-local-memcached=never with an external memcached "
                "instance"
            )
        return [
            memcached_bin,
            "-u",
            "root",
            "-l",
            self.memcached_host,
            "-p",
            str(self.memcached_port),
            "-c",
            "10000",
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
        return "[RDMA-DBG] Server polling thread ready" in line

    def _monitor(self, global_id, pipe):
        for raw_line in iter(pipe.readline, ""):
            line = raw_line.rstrip()
            if self.verbose:
                print(f"[petps_server:{global_id}] {line}")
            if self.is_ready_line(line):
                ready = self.ready_threads.setdefault(global_id, set())
                ready.add(line.rsplit(" ", 1)[-1])
                if len(ready) >= self.thread_num:
                    self.ready.add(global_id)

    def _start_memcached(self):
        cmd = self.build_memcached_cmd()
        self.memcached_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        print(
            f"[petps-memcached] started with pid={self.memcached_process.pid} "
            f"host={self.memcached_host} port={self.memcached_port}"
        )

        time.sleep(0.2)
        if self.memcached_process.poll() is not None:
            log = ""
            if self.memcached_process.stdout is not None:
                log = self.memcached_process.stdout.read()
            err = (
                f"memcached exited early with code {self.memcached_process.returncode}"
            )
            if log:
                err += f"\nmemcached output:\n{log}"
            raise RuntimeError(err)

    def check_memcached_ready(self):
        last_error = None
        for attempt in range(1, self.memcached_check_retries + 1):
            try:
                with socket.create_connection(
                    (self.memcached_host, self.memcached_port),
                    timeout=self.memcached_check_timeout,
                ) as sock:
                    sock.settimeout(self.memcached_check_timeout)
                    sock.sendall(b"get serverNum\r\n")
                    data = sock.recv(4096)
                    if b"END\r\n" in data or b"VALUE serverNum" in data:
                        self.emit_status(
                            "memcached-ready",
                            f"attempt={attempt} host={self.memcached_host}:{self.memcached_port}",
                        )
                        return
                    last_error = RuntimeError(
                        "memcached responded but without expected get reply"
                    )
            except OSError as exc:
                last_error = exc
            self.emit_status(
                "memcached-wait",
                f"attempt={attempt}/{self.memcached_check_retries} "
                f"host={self.memcached_host}:{self.memcached_port}",
            )
            time.sleep(0.2)

        raise RuntimeError(
            "memcached is not reachable or not ready at "
            f"{self.memcached_host}:{self.memcached_port}; "
            "set --use-local-memcached=always|auto|never and "
            "RECSTORE_MEMCACHED_HOST/RECSTORE_MEMCACHED_PORT as needed"
        ) from last_error

    def reset_memcached_state(self):
        payload = (
            b"flush_all\r\n"
            b"set serverNum 0 0 1\r\n0\r\n"
            b"set clientNum 0 0 1\r\n0\r\n"
            b"set xmh-consistent-dsm 0 0 1\r\n1\r\n"
            b"get serverNum\r\n"
            b"get clientNum\r\n"
            b"get xmh-consistent-dsm\r\n"
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
            if (
                b"OK\r\n" not in response
                or response.count(b"STORED\r\n") < 3
                or b"VALUE serverNum 0 1\r\n0\r\n" not in response
                or b"VALUE clientNum 0 1\r\n0\r\n" not in response
                or b"VALUE xmh-consistent-dsm 0 1\r\n1\r\n" not in response
            ):
                raise RuntimeError(
                    "failed to initialize memcached state; "
                    f"response was: {response!r}"
                )
            self.emit_status(
                "memcached-reset",
                f"host={self.memcached_host}:{self.memcached_port}",
            )

    def _prepare_memcached(self):
        if self.use_local_memcached not in {"always", "auto", "never"}:
            raise ValueError(
                "use_local_memcached must be one of: always, auto, never"
            )

        if self.use_local_memcached == "always":
            self._start_memcached()
            self.check_memcached_ready()
            self.reset_memcached_state()
            return

        try:
            self.check_memcached_ready()
            self.reset_memcached_state()
            return
        except RuntimeError:
            if self.use_local_memcached != "auto":
                raise

        # Existing memcached may be unreachable or incompatible with the reset
        # sequence we require. Fall back to a dedicated local memcached
        # instance on a fresh port.
        self.memcached_host = "127.0.0.1"
        self.memcached_port = self._allocate_local_memcached_port()
        self._start_memcached()
        self.check_memcached_ready()
        self.reset_memcached_state()

    def start(self):
        if not self.server_path.exists():
            raise FileNotFoundError(f"Server binary not found: {self.server_path}")

        self._prepare_memcached()
        env = self.build_env()
        print(
            "[petps-memcached] server env "
            f"host={env['RECSTORE_MEMCACHED_HOST']} "
            f"port={env['RECSTORE_MEMCACHED_PORT']}"
        )

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

        # PetPS servers cannot reach polling-ready until the client joins the
        # DSM-init barrier. If no post-barrier ready line appears yet, let the
        # client proceed as long as the server process is still alive; the RDMA
        # client waits on a memcached server-ready key before sending RPCs.
        if not self.ready:
            for global_id, (process, _thread) in enumerate(self.processes):
                if process.poll() is None:
                    self.ready.add(global_id)

        deadline = time.time() + self.timeout
        next_refresh = time.time() + self.status_refresh_interval
        while len(self.ready) < self.num_servers:
            if time.time() > deadline:
                self.emit_status("startup-timeout", f"timeout={self.timeout}s")
                self.stop()
                raise TimeoutError(
                    f"Timed out waiting for {self.num_servers} petps_server processes"
                )
            for process, _thread in self.processes:
                if process.poll() is not None:
                    self.emit_status(
                        "startup-crash",
                        f"exit_code={process.returncode}",
                    )
                    self.stop()
                    raise RuntimeError(
                        f"petps_server exited early with code {process.returncode}"
                    )
            if (
                self.status_refresh_interval > 0
                and time.time() >= next_refresh
            ):
                self.emit_status("startup-wait")
                next_refresh = time.time() + self.status_refresh_interval
            time.sleep(0.2)

    def run_client(self, argv, client_index=0, stream_output=True, timeout=None):
        cmd = self.build_client_cmd(argv, client_index=client_index)
        env = self.build_env()
        print(
            "[petps-memcached] client env "
            f"host={env['RECSTORE_MEMCACHED_HOST']} "
            f"port={env['RECSTORE_MEMCACHED_PORT']} "
            f"client_index={client_index}"
        )
        if not stream_output:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                check=False,
                env=env,
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
            env=env,
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
