# RecStore `src/ps` 参数服务器（Parameter Server）模块总结

本文档基于当前仓库中 `src/ps` 目录下的源码整理，说明该模块的职责、对外接口形态与典型使用方式。

---

## 1. 模块定位与总体职责

`src/ps` 实现了一套**分布式/单机可切换的嵌入与稀疏参数服务**，用于：

- 按 **uint64 主键** 存储与读取 **变长维度 float 向量**（每个 key 可对应不同 embedding 维度）；
- 通过 **Put / Get** 批量读写参数；
- 通过 **UpdateParameter** 将梯度写回并与服务端 **Optimizer**（当前实现中为 SGD）联动更新；
- 通过 **InitEmbeddingTable** 在服务端初始化逻辑表（行数、维度）并挂接优化器；
- 通过 **Command** 下发管理类指令（清空、重载、加载假数据等）。

底层存储由 `CachePS` 抽象：根据 JSON 中的 `base_kv_config` 解析并构造 `BaseKV`（KV 引擎来自 `storage/kv_engine`，与 RecStore 主工程其他部分共享）。

协议层同时支持 **gRPC** 与 **bRPC**（两套几乎同构的 `.proto`），可由全局配置 `cache_ps.ps_type` 在 **`GRPC` / `BRPC`** 之间切换。

---

## 2. 目录结构概览

| 路径 | 作用 |
|------|------|
| `proto/` | `ps.proto`（gRPC）、`ps_brpc.proto`（bRPC，`cc_generic_services = true`） |
| `base/` | `BasePSClient`、`BaseParameterServer`、`CachePS`、参数压缩结构 `parameters.*`、`Postoffice`、`ShardManager`、`config` 等 |
| `grpc/` | gRPC 版服务端 `GRPCParameterServer`、客户端 `GRPCParameterClient`、`DistributedGRPCParameterClient` |
| `brpc/` | bRPC 版服务端 `BRPCParameterServer`、`BRPCParameterServiceImpl`、客户端 `BRPCParameterClient`、`DistributedBRPCParameterClient` |
| `rdma/` | 基于 RDMA / DSM 的另一套客户端（`PetPSClient` 等），接口基类为 `rdma/base_client.h` 中的 `BaseParameterClient`，与 `ps/base/base_client.h` 的 `BasePSClient` **不是同一套抽象** |
| `python_client/` | Python 测试/工具脚本（部分依赖缺失的 `client` 模块，需与工程内 PyTorch/gRPC 绑定配合使用） |
| `ps_server.cpp` | **统一入口**：按配置选择 gRPC 或 bRPC 的 `BaseParameterServer` 实现并 `Run()` |
| `provider.h` | 通用 FIFO / 优先级任务队列模板（与 PS 核心链路弱相关，可供调度使用） |

---

## 3. 对外 RPC 接口（Protocol Buffers）

包名：
 
 
- gRPC：`recstoreps`（`ps.proto`）
- bRPC：`recstoreps_brpc`（`ps_brpc.proto`）

服务名均为 **`ParameterService`**，RPC 如下。

### 3.1 `GetParameter`

- **请求**：`keys`（bytes，按客户端约定编码的 key 序列）、可选 `model_name`、可选 `perf`（性能统计开关）。
- **响应**：`parameter_value`（bytes，**压缩后的** `ParameterCompressItem` 批）、`keys`（可选回传）。

服务端在 `ParameterServiceImpl` / `BRPCParameterServiceImpl` 中调用 `CachePS::GetParameterRun2Completion`，再用 `ParameterCompressor` 打成单块 `blocks[0]` 返回。

### 3.2 `PutParameter`

- **请求**：`parameter_value`（bytes，同样为压缩格式批次）。
- **响应**：`sucess`（拼写保持与 proto 一致）。

对应 `CachePS::PutParameter` → `BaseKV::BatchPut`。

### 3.3 `UpdateParameter`

- **请求**：`table_name`、`gradients`（压缩批次）。
- **响应**：`success`。

对应 `CachePS::UpdateParameter` → `Optimizer::Update`（需事先 `InitTable`）。

### 3.4 `InitEmbeddingTable`

- **请求**：`table_name`、`config_payload`（JSON 序列化字符串，含 `num_embeddings`、`embedding_dim`，见 `EmbeddingTableConfig::Serialize()`）。
- **响应**：`success`。

对应 `CachePS::InitTable`。

### 3.5 `Command`

- **请求**：`command` 枚举 `CLEAR_PS`、`RELOAD_PS`、`LOAD_FAKE_DATA`，以及多组 `arg1/arg2/arg3`（bytes 向量）。
- **响应**：`reply`（文本）。

用于清空 KV、加载检查点/假数据等运维类操作（具体参数由服务端 `Command` 实现解析）。

---

## 4. C++ 客户端抽象：`recstore::BasePSClient`

定义于 `base/base_client.h`，是所有 **gRPC/bRPC 统一风格** 客户端的基类。

主要虚函数（节选）：

- `GetParameter(ConstArray<uint64_t> keys, float* values)`：同步拉取，结果写入**调用方提供的连续缓冲区**（需与 key 顺序、各向量维度约定一致）。
- `PutParameter` / `UpdateParameter` / `UpdateParameterFlat`：写参数或按表名更新梯度（Flat  variant 避免 `vector<vector<float>>` 开销）。
- `InitEmbeddingTable`：在服务端建表。
- `AsyncGetParameter`：异步读（实现因 transport 而异）。
- `Command(PSCommand)`：管理命令。
- **Prefetch 系列**：`PrefetchParameter` → `IsPrefetchDone` / `WaitForPrefetch` / `GetPrefetchResult` / `GetPrefetchResultFlat`，用于流水线重叠。

`EmbeddingTableConfig` 含 `num_embeddings`、`embedding_dim`，`Serialize()` 与 `InitEmbeddingTableRequest.config_payload` 对齐。

---

## 5. 具体客户端实现与工厂注册

### 5.1 单机：`GRPCParameterClient` / `BRPCParameterClient`

- **头文件**：`grpc/grpc_ps_client.h`、`brpc/brpc_ps_client.h`
- **JSON 构造**常用字段：`host`（默认 `localhost`）、`port`（gRPC 默认 `15000`）、`shard`
- **gRPC**：可多 stub（`FLAGS_get_parameter_threads`）；支持异步批量 Prefetch、Prewrite（`EmbWriteAsync` 等）。
- **bRPC**：`timeout_ms`、`max_retry` 等由 gflags / 配置控制；大块数据可走 `Controller::response_attachment()`（见 `ExtractGetResponseReader`）。

**工厂注册**（`base::Factory<BasePSClient, json>`）：

| 注册键 | 类型 |
|--------|------|
| `"grpc"` | `GRPCParameterClient` |
| `"brpc"` | `BRPCParameterClient` |

### 5.2 分布式：`DistributedGRPCParameterClient` / `DistributedBRPCParameterClient`

- **头文件**：`grpc/dist_grpc_ps_client.h`、`brpc/dist_brpc_ps_client.h`
- **配置**：顶层 JSON 需包含 **`distributed_client`** 对象，字段包括：
  - `servers`：数组，每项 `{ "host", "port", "shard" }`
  - `num_shards`：整数
  - 可选 `max_keys_per_request`（默认 500）、`hash_method`（默认 `"city_hash"`）
- 行为：对 key 做 hash 分区，将请求拆到多个单机客户端，再合并结果。

**工厂注册键**：

| 注册键 | 类型 |
|--------|------|
| `"distributed_grpc"` | `DistributedGRPCParameterClient` |
| `"distributed_brpc"` | `DistributedBRPCParameterClient` |

说明：`src/framework/op.cc` 中的 `create_ps_client_from_config` 目前对 `distributed_client` 分支写了 `if (false && ...)`，**默认不会走分布式工厂**；若要在主框架里启用，需改该逻辑或自行 `Factory::NewInstance("distributed_grpc", config)`。

---

## 6. 服务端：`BaseParameterServer` 与 `ps_server`

### 6.1 抽象基类

`base/base_ps_server.h`：`Init(const json&)`、`Run()`。

### 6.2 两种实现

- **`GRPCParameterServer`**（`grpc/grpc_ps_server.cpp`）
- **`BRPCParameterServer`**（`brpc/brpc_ps_server.cpp`）

均通过 `FACTORY_REGISTER(BaseParameterServer, ...)` 注册，名称为类名字符串。

### 6.3 `ps_server` 可执行程序（`ps_server.cpp`）

- 使用 `FLAGS_config_path`，若文件不存在则回退 `FLAGS_brpc_config_path`。
- 读取 JSON 后，从 **`cache_ps.ps_type`** 读取 `"GRPC"` 或 `"BRPC"`（大小写不敏感），再：

  - `GRPC` → `Factory<BaseParameterServer>::NewInstance("GRPCParameterServer")`
  - `BRPC` → `...("BRPCParameterServer")`

- 调用 `Init(config)`、`Run()`。

### 6.4 监听与分片

- **gRPC 单进程默认**：`0.0.0.0:15000`（硬编码于 `GRPCParameterServer::Run` 单分片分支）。
- **gRPC 多分片**：`cache_ps.num_shards > 1` 且提供 `cache_ps.servers` 数组，**每个 shard 一线程**各自 `CachePS` + `ServerBuilder`，可按 shard 改写 `base_kv_config.path` 后缀。
- **bRPC 单分片**：`0.0.0.0:` + `FLAGS_brpc_server_port`。
- **bRPC 多分片**：类似多线程多监听地址；注意当前实现里各 shard 共用同一份 `config_["cache_ps"]` 构造 `CachePS`（与 gRPC 分片对 path 的处理不完全一致，部署时需自行核对数据目录是否冲突）。

---

## 7. 服务端核心存储：`CachePS`

定义于 `base/cache_ps_impl.h`。

- 构造函数从 `cache_ps` JSON 读 `num_threads`、`base_kv_config`，经 `ResolveEngine` + `Factory<BaseKV>` 创建底层 KV。
- **读**：`GetParameterRun2Completion` → `BaseKV::BatchGet`，填充 `ParameterPack`（key + dim + float*）。
- **写**：`PutParameter` → `BatchPut`（支持协程 `push_type&` 重载）。
- **embedding 训练路径**：`InitTable` 创建默认 `SGD(0.01)` 与 `EmbeddingTableConfig`；`UpdateParameter` 将 `ParameterCompressReader` 交给 `optimizer_->Update`。

---

## 8. 数据编码：`ParameterCompressItem`

定义于 `base/parameters.h`：`#pragma pack` 结构为 `uint64_t key`、`int dim`、`float embedding[0]`，配合 `FlatItemCompressor` / `ParameterCompressor` 将多条记录打成单字节块在网络上传输。客户端与服务端必须**使用同一套压缩布局**。

---

## 9. 典型使用方式

### 9.1 启动参数服务器

1. 准备全局 JSON（示例字段见工程内 `recstore_config.json` 或测试配置），其中包含 **`cache_ps`** 段：`ps_type`、`num_threads`、`base_kv_config` 等。
2. 运行 `ps_server`，并指定 `--config_path`（或可用的 `brpc_config_path`）。

### 9.2 C++ 客户端（单机）

```cpp
#include "base/factory.h"
#include "ps/base/base_client.h"

json cfg = {{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}};
std::unique_ptr<recstore::BasePSClient> client(
    base::Factory<recstore::BasePSClient, json>::NewInstance("grpc", cfg));
// 或 "brpc"
```

参考测试：`brpc/brpc_ps_client_test.cpp`（Factory + Put/Get/Clear）。

### 9.3 C++ 客户端（分布式）

```cpp
json cfg = {
    {"distributed_client",
     {{"servers",
       {{{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}},
        {{"host", "127.0.0.1"}, {"port", 15001}, {"shard", 1}}}},
      {"num_shards", 2},
      {"hash_method", "city_hash"}}}};
auto client = base::Factory<recstore::BasePSClient, json>::NewInstance(
    "distributed_grpc", cfg);
```

参考：`grpc/dist_grpc_ps_client_test.cpp`。

### 9.4 与主框架集成

`create_ps_client_from_config`（`src/framework/op.cc`）根据 `cache_ps.ps_type` 选择 **`grpc` 或 `brpc`**，并从 `client` 子对象或默认 `127.0.0.1:15000` 构造**单机**客户端。

---

## 10. RDMA 子目录 (`rdma/`)

- `PetPSClient` 等实现 `petps::BaseParameterClient`（定义在 `rdma/base_client.h`），依赖 DSM（`third_party/Mayfly-main`），走 RDMA 路径而非 gRPC/bRPC `ParameterService`。
- 工厂键为 `"PetPSClient"`（与 `BasePSClient` 的 `"grpc"`/`"brpc"` 不同族）。
- 适用于低延迟、与 **PetPS / AllShards** 相关的实验或部署；与 `ps_server` + Protobuf 路线**并行存在**，不要混用同一客户端基类。

---

## 11. Python 客户端 (`python_client/`)

包含 `DistTensor`、`DistEmb`、`EmbBag`、`load_client.py` 等脚本与样例配置。部分文件引用 `from client import GRPCParameterClient` 等**工程外或生成模块**；在无对应 pybind/扩展时可能无法直接运行。它们表达的是**产品化用法意图**（PyTorch 张量 → PS），但集成状态需以当前构建脚本为准。

---

## 12. 依赖与可选编译特性

- **Proto**：生成 `ps.pb` / `ps.grpc.pb` / `ps_brpc.pb`（见 `proto/CMakeLists.txt`）。
- **Folly、gflags、glog**：服务端与部分客户端初始化。
- **可选**：`ENABLE_PERF_REPORT`（上报埋点）、`USE_BRPC_CPU_PROFILER`、`ENABLE_GPERF_PROFILING`（`ps_server` CMake 中配置）。

---

## 13. 小结

| 维度 | 要点 |
|------|------|
| **做什么** | 基于 KV 的稀疏/嵌入参数服务 + 可选 SGD 更新 + 运维 Command |
| **对外协议** | `ParameterService`（gRPC / bRPC 双实现，消息结构对齐） |
| **C++ 主接口** | `BasePSClient` + 单机/分布式实现 + `Factory` 字符串键 |
| **服务端入口** | `ps_server`，由 `cache_ps.ps_type` 选择 gRPC 或 bRPC 的 `BaseParameterServer` |
| **存储** | `CachePS` → `BaseKV`（JSON 驱动选型） |

若后续需要**单独维护架构图或 OpenAPI 式文档**，可在本文基础上增补配置字段的完整 JSON Schema 与各 RPC 的字段级说明（直接引用 `ps.proto` / `ps_brpc.proto` 即可作为权威定义）。
