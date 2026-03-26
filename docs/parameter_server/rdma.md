# RDMA 模块

???+ warning "注意"
    该模块当前处于实验阶段，不建议用于生产环境。

## 概述

RDMA 模块是基于 Mayfly 分布式共享内存系统的实现。

## 文件组织

| 文件 | 说明 |
|------|------|
| `petps_server.cc` | RDMA 参数服务器 |
| `petps_client.h` / `.cc` | RDMA 客户端 |
| `allshards_ps_client.h` / `.cc` | 多分片客户端 |
| `petps_magic.h` | 调试接口 |
| `base_client.h` | 客户端基类 |
