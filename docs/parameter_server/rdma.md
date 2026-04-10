# RDMA 模块

!!! warning "边界说明"
    RDMA 当前是 `BaseParameterClient` 路线，不是 `BasePSClient` 路线，不能直接通过 `KVClientOp` 的 `ps_type` 切换进去。

## 定位

RDMA 模块基于 Mayfly / DSM，提供独立的高性能 `Get/Put` 数据面。

它与 GRPC / BRPC 的“靠齐”主要体现在：

- 配置角色清晰
- 测试入口清晰
- runbook 清晰

而不体现在强行接入主框架抽象。

## 当前完成范围

- 单分片 `Get/Put`
- 多分片 allshards 路由与拆分
- 独立 `petps_server` 入口
- 单分片 / 多分片 integration test
- transport benchmark 驱动

## 配置角色

RDMA 专项配置位于：

- `src/test/scripts/recstore_config.rdma_test.json`
- `src/test/scripts/recstore_config.rdma_multishard_test.json`

这些配置只服务 RDMA 专项验证，不应回落到根目录默认 `recstore_config.json`。

## 相关入口

### 单分片 integration

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/scripts/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.MissingKeysReturnZeroSlots \
  --client-timeout 60 \
  --use-local-memcached=never
```

### 多分片 integration

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 2 \
  --config-path ./src/test/scripts/recstore_config.rdma_multishard_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard \
  --client-timeout 60 \
  --use-local-memcached=never
```

### transport benchmark

```bash
python3 src/test/scripts/run_rdma_transport_benchmarks.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --use-local-memcached=auto
```

## 明确非目标

- 不直接并入统一 `ps_server`
- 不直接实现 `BasePSClient`
- 不通过 `KVClientOp` 自动切到 RDMA
