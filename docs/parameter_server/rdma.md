# RDMA 模块

???+ warning "注意"
    RDMA 当前是 `BaseParameterClient` 路线，不是 `BasePSClient` 路线，不能直接通过 `KVClientOp` 的 `ps_type` 切换进去。

## 概述

RDMA 模块基于 Mayfly/DSM，提供独立的高性能 `Get/Put` 数据面。

## 当前完成范围

- 单分片 `Get/Put`
- 多分片 allshards 路由与拆分
- 独立 `petps_server` 入口
- 集成测试脚本与 benchmark 驱动

## 明确非目标

- 不直接并入统一 `ps_server`
- 不直接实现 `BasePSClient`
- 不通过 `KVClientOp` 自动切到 RDMA
