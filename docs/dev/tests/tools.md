# 测试用工具

## ps_server_helpers.py

`src/test/scripts/ps_server_helpers.py` 模块提供了一系列辅助函数，用于在测试环境中查找、检测和配置 `ps_server`。它通常与 `ps_server_runner.py` 配合使用，用于自动化测试流程。

| 变量名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PS_SERVER_PATH` | 指定 `ps_server` 的绝对路径 | 自动查找 |
| `RECSTORE_CONFIG` | 指定 `recstore_config.json` 路径 | None |
| `PS_LOG_DIR` | 指定日志输出目录 | `./logs` |
| `PS_TIMEOUT` | 启动超时时间（秒） | 60 |
| `PS_NUM_SHARDS` | 期望的分片数量 | 2 |
| `NO_PS_SERVER` | 强制跳过启动服务器 (Set to '1' or 'true') | False |

### `find_ps_server_binary()`

在 `build/bin/ps_server` 等常见构建目录中查找可执行文件。

| | 说明 |
| :--- | :--- |
| **返回值** | `ps_server` 的绝对路径。 |
| **查找顺序** | `OS ENV(PS_SERVER_PATH)` -> `./bin` -> `./build/bin` -> 上级目录的构建路径。 |

### `check_ps_server_running(ports=None)`

检查 Parameter Server 的默认端口（或指定端口）是否已在监听。

| | 说明 |
| :--- | :--- |
| **参数** | `ports` (list, optional): 要检查的端口列表。默认为 `[15000, 15001, 15002, 15003]`。 |
| **返回值** | `(is_running: bool, open_ports: list)` |

### `should_skip_server_start()`

判断当前测试是否应该跳过启动 `ps_server` 的步骤。判断逻辑:

- 检查环境变量 `CI` 或 `GITHUB_ACTIONS`。
- 检查环境变量 `NO_PS_SERVER`。
- 检查端口是否已被占用（意味着服务已经在运行）。

返回值: `(skip: bool, reason: str)`

### `get_server_config()`

获取标准化的服务器配置字典，优先读取环境变量。

返回值字典:

| 键名 | 说明 |
| :--- | :--- |
| `server_path` | 二进制文件路径 |
| `config_path` | 配置文件路径 (`RECSTORE_CONFIG`) |
| `log_dir` | 日志目录 (`PS_LOG_DIR`) |
| `timeout` | 超时时间 (`PS_TIMEOUT`) |
| `num_shards` | 分片数量 (`PS_NUM_SHARDS`) |

