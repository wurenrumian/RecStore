#!/usr/bin/env python3

import os
import socket


def find_ps_server_binary():
    """Find ps_server binary in common locations."""
    server_path = os.environ.get('PS_SERVER_PATH')
    if server_path:
        return os.path.abspath(server_path)
    
    candidates = [
        './bin/ps_server',
        './build/bin/ps_server',
        '../bin/ps_server',
        '../../build/bin/ps_server',
        '../../../build/bin/ps_server',
        '../../../../build/bin/ps_server',
    ]
    
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate
    
    return os.path.abspath('./build/bin/ps_server')


def is_port_open(host, port, timeout=1):
    """Check if a port is open/listening."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_ps_server_running(ports=None):
    """Check if ps_server is running by checking if ports are open."""
    if ports is None:
        ports = [15000, 15001, 15002, 15003]
    
    open_ports = [port for port in ports if is_port_open('127.0.0.1', port)]
    
    if open_ports:
        return True, open_ports
    return False, []


def should_skip_server_start():
    """Determine if we should skip starting ps_server."""
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    no_server = os.environ.get('NO_PS_SERVER', '').lower() in ('1', 'true', 'yes')
    
    if is_ci or no_server:
        return True, "CI" if is_ci else "NO_PS_SERVER"
    
    running, ports = check_ps_server_running()
    if running:
        return True, f"already_running:{ports}"
    
    return False, None


def get_server_config():
    """Get server configuration from environment."""
    return {
        'server_path': find_ps_server_binary(),
        'config_path': os.environ.get('RECSTORE_CONFIG'),
        'log_dir': os.environ.get('PS_LOG_DIR', './logs'),
        'timeout': int(os.environ.get('PS_TIMEOUT', '60')),
        'num_shards': int(os.environ.get('PS_NUM_SHARDS', '2')),
    }
