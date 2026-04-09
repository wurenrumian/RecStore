# gRPC 模块讲解（`src/ps/grpc`）

## 1. 模块定位

`src/ps/grpc` 是当前 PS 的主干 RPC 实现之一，接口完整、与 `BasePSClient`/`BaseParameterServer` 抽象对齐，且已接入主框架调用链。

---

## 2. 组成部分

- `grpc_ps_server.h/.cpp`：`GRPCParameterServer` + RPC 服务实现
- `grpc_ps_client.h/.cpp`：单节点客户端 `GRPCParameterClient`
- `dist_grpc_ps_client.h/.cpp`：分布式客户端（按 hash 路由到 shard）
- `grpc_ps_client_pytorch.cpp`：PyTorch 绑定
- `*_test.cpp`：客户端与分布式客户端测试

---

## 3. 架构要点

- 对外 RPC：`Get/Put/Update/InitEmbeddingTable/Command`
- 服务端 handler 统一收敛到 `CachePS`
- 客户端支持批量分包、Prefetch、Flat Update、FakeData 带宽压测
- 多分片服务端支持按 shard 启多线程实例

---

## 4. 你可把它当作 RDMA 的“功能对齐基线”

如果要让 RDMA 达到 grpc 同等可用度，至少要做到：

1. 接口能力对齐（读/写/更新/初始化/命令）
2. 与主框架 `KVClientOp` 调用语义可衔接
3. 具备完整测试入口（单机+分片）

