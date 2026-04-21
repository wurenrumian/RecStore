# RDMA 模块

## 概述

!!! warning "边界说明"
    RDMA 当前是 `BaseParameterClient` 路线，不是 `BasePSClient` 路线，不能直接通过 `KVClientOp` 的 `ps_type` 切换进去。

RDMA 模块基于 Mayfly/DSM，提供独立 `petps_server` 数据面。  
本页聚焦“怎么启动、怎么测、怎么排障”，不展开实现细节。

## 配置

RDMA 专项配置位于：

| 配置文件 | 用途 |
|----------|------|
| `src/test/configs/recstore_config.rdma_test.json` | 单分片测试 |
| `src/test/configs/recstore_config.rdma_multishard_test.json` | 多分片测试 |

!!! note
    这些配置只服务 RDMA 专项验证，不应回落到根目录默认 `recstore_config.json`。

配置示例 (`recstore_config.rdma_test.json`)：

```json
{
  "cache_ps": {
    "ps_type": "GRPC",
    "max_batch_keys_size": 65536,
    "num_threads": 32,
    "num_shards": 1,
    "servers": [{"host": "127.0.0.1", "port": 25000, "shard": 0}],
    "base_kv_config": {
      "path": "/tmp/recstore_data_rdma_test",
      "capacity": 1000000,
      "value_size": 16,
      "value_type": "DRAM",
      "value_memory_management": "PersistLoopShmMalloc"
    }
  },
  "distributed_client": {
    "num_shards": 1,
    "hash_method": "city_hash",
    "max_keys_per_request": 500,
    "servers": [{"host": "127.0.0.1", "port": 25000, "shard": 0}]
  }
}
```

## 测试入口

!!! note
    本节命令默认在仓库根目录执行（`/app/RecStore`）。
    如果当前目录是 `build/`，请将脚本路径改为 `../src/test/scripts/...`，
    并将二进制路径改为 `./bin/...`。

### 启动 memcached

```bash
memcached -u root -l 127.0.0.1 -p 21211 -c 10000 -vv
```

`run_petps_integration.py` 的 `--use-local-memcached` 参数控制 memcached
来源：

| 参数值 | 行为 |
|--------|------|
| `auto` | 先尝试外部 memcached；如果不可用或 reset 失败，则启动系统 `memcached` 二进制（推荐默认模式） |
| `never` | 只使用已在 `127.0.0.1:21211` 启动的外部 memcached，适用于你需要严格绑定外部 memcached 的场景 |
| `always` | 直接启动系统 `memcached` 二进制 |

!!! note
    `auto` / `always` 依赖机器上已安装真实 `memcached` 命令，不再使用 Python fake memcached。
    `run_petps_integration.py` 与 `run_rdma_transport_benchmarks.py` 默认会隐藏
    memcached 相关噪音日志；如需查看完整 runner 日志可追加
    `--show-runner-logs`。
    实际使用中建议优先使用 `--use-local-memcached=auto`，让脚本统一管理
    memcached 生命周期并减少环境差异。

### RDMA Server 启动（推荐）

为降低 `petps_server` 直接启动时的参数复杂度（`global_id` /
`num_server_processes` / `num_client_processes` 等），推荐使用：

```bash
python3 src/test/scripts/run_petps_server.py \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --use-local-memcached=auto
```

如需调优，也可追加：
`--rdma-per-thread-response-limit-bytes`、
`--rdma-server-ready-timeout-sec`、
`--rdma-server-ready-poll-ms`、
`--rdma-client-receive-arena-bytes`、
`--validate-routing`。

该入口会：

- 根据配置推断 `server-count`（也可显式传 `--server-count`）
- 自动注入 RDMA 所需运行参数
- 统一处理 memcached（`auto/always/never`）

### 单分片 integration

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.MissingKeysReturnZeroSlots \
  --client-timeout 15 \
  --use-local-memcached=auto
```

### 多分片 integration

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 2 \
  --config-path ./src/test/configs/recstore_config.rdma_multishard_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard \
  --client-timeout 20 \
  --use-local-memcached=auto
```

### transport benchmark

```bash
python3 src/test/scripts/run_rdma_transport_benchmarks.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --iterations 20 \
  --rounds 50 \
  --rdma-warmup-rounds 5 \
  --use-local-memcached=auto
```

!!! note
    `run_petps_integration.py` 对 `client-timeout` 与 `cluster-timeout` 采用 15 秒硬上限；超过 15 秒会自动终止并清理进程。

### ctest 入口（可选）

当构建时启用 `ENABLE_RDMA_INTEGRATION_TESTS=ON`，可直接运行：

```bash
ctest --test-dir ./build -R "petps_single_shard_test|petps_multi_shard_test" -VV
```

### Op-layer 验证

当 `cache_ps.ps_type` 设置为 `RDMA` 时，framework op layer 会通过
`RDMAPSClientAdapter` 复用 PetPS/RDMA 数据面。可使用现有 PyTorch client
测试验证该配置切换路径：

```bash
ctest --test-dir ./build -R "^pytorch_client_test_rdma_auto$" -VV
```

两个测试均使用 `src/test/configs/recstore_config.op_rdma.json`，覆盖
init、write、read、update 与 prefetch 正确性。

如需手工切换 memcached 策略，也可在运行前设置环境变量：

```bash
export RECSTORE_USE_LOCAL_MEMCACHED=auto   # 或 always / never
ctest --test-dir ./build -R "^pytorch_client_test_rdma$" -VV
```

其中：

- `auto`：优先尝试外部 memcached，失败时自动拉起本地 `memcached` 二进制
- `never`：只使用外部 `127.0.0.1:21211` 的 memcached
- `always`：直接拉起本地 `memcached` 二进制


## 稳定性机制（当前默认行为）

`run_petps_integration.py` / `PetPSClusterRunner` 已内置以下稳定性机制：

1. 每次测试启动前自动重置 memcached 状态：
   - `flush_all`
   - 重建 `serverNum=0`、`clientNum=0`、`xmh-consistent-dsm=1`
   - 通过 `get` 回读校验三项键值
2. 对 memcached 就绪进行重试探测，失败会报明确错误。
3. 启动阶段周期打印状态刷新日志（`[petps-status]`）。
4. `client-timeout` 与 `cluster-timeout` 统一受 15 秒硬上限保护，超时自动清理子进程。

典型状态日志：

- `phase=memcached-ready`
- `phase=memcached-reset`
- `phase=startup-wait`
- `phase=startup-timeout`
- `phase=startup-crash`

## 排障 Runbook

### 常见症状：偶发卡住（非持续高频连接）

建议按以下顺序排查：

1. 先看状态日志停在什么阶段：
   - 若停在 `memcached-wait`：优先检查 memcached 可达性。
   - 若停在 `startup-wait`：优先检查 `petps_server` 是否已启动且未崩溃。
2. 检查 21211 端口连接关系：

```bash
ss -tnp | grep ':21211'
lsof -nP -iTCP:21211
fuser -v 21211/tcp
```

3. 如需手工验证 memcached 重置，可执行（与脚本内置逻辑一致）：

```bash
printf 'flush_all\r\nset serverNum 0 0 1\r\n0\r\nset clientNum 0 0 1\r\n0\r\nset xmh-consistent-dsm 0 0 1\r\n1\r\nget serverNum\r\nget clientNum\r\nget xmh-consistent-dsm\r\nquit\r\n' | nc -q 1 127.0.0.1 21211
```

4. 多分片场景优先确认：
   - `recstore_config.rdma_multishard_test.json` 中 `num_shards/servers` 与测试参数一致。
   - 本地没有残留旧 `petps_server` 进程占用同一组资源。

## 关键入口文件

- `src/test/scripts/run_petps_server.py`
- `src/test/scripts/run_petps_integration.py`
- `src/test/scripts/run_rdma_transport_benchmarks.py`
- `src/ps/rdma/petps_server.cc`
- `src/ps/rdma/petps_client.cc`
