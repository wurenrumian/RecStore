# RecStore 配置文档

## 配置文件结构

RecStore 配置采用 JSON 格式，位于根目录，包含三个主要部分：`cache_ps`、`distributed_client` 和 `client`。

## 1. cache_ps 配置

`cache_ps` 配置用于参数服务器（Parameter Server）端，定义服务器的运行参数和底层存储引擎。

### 1.1 服务器基础配置

| 字段 | 类型 | 必填 | 说明 | 作用位置 |
|------|------|------|------|----------|
| `ps_type` | string | 是 | 参数服务器类型，可选 `"GRPC"` 或 `"RDMA"` | [src/base_ps/cache_ps_impl.h](../src/base_ps/cache_ps_impl.h)、[src/grpc_ps/grpc_ps_server.cpp](../src/grpc_ps/grpc_ps_server.cpp)、[src/rdma_ps/petps_server.cc](../src/rdma_ps/petps_server.cc)，用于决定服务器通信协议 |
| `max_batch_keys_size` | integer | 是 | 单次批量请求的最大键数量 | [src/grpc_ps/grpc_ps_client.h](../src/grpc_ps/grpc_ps_client.h)，控制客户端批量请求的上限，避免单次请求过大 |
| `num_threads` | integer | 是 | 服务器工作线程数 | [src/base_ps/cache_ps_impl.h](../src/base_ps/cache_ps_impl.h)、[src/storage/kv_engine/base_kv.h](../src/storage/kv_engine/base_kv.h)，传递给底层 KV 引擎的 `BaseKVConfig.num_threads_`，影响并发处理能力 |
| `num_shards` | integer | 是 | 分片数量，应等于 `servers` 数组长度 | [src/base_ps/cache_ps_impl.h](../src/base_ps/cache_ps_impl.h)，用于分布式部署时的数据分片 |
| `servers` | array | 是 | 服务器节点配置数组 | 多个文件使用，定义所有参数服务器节点 |

### 1.2 servers 数组配置

每个服务器节点包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `host` | string | 是 | 服务器主机地址 |
| `port` | integer | 是 | 服务器监听端口 |
| `shard` | integer | 是 | 分片编号，从 0 开始 |

### 1.3 base_kv_config 配置

`base_kv_config` 定义底层键值存储引擎的配置，由 [src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h) 的 `ResolveEngine` 函数根据 `index_type` 和 `value_type` 自动推导引擎类型。

#### 通用必填字段（非 HYBRID 模式）

| 字段 | 类型 | 必填 | 说明 | 作用位置 |
|------|------|------|------|----------|
| `path` | string | 是 | 存储路径，建议每个实例使用独立目录 | [src/storage/kv_engine/base_kv.h](../src/storage/kv_engine/base_kv.h)，用于持久化存储的工作目录 |
| `index_type` | string | 是 | 索引存储类型：`"DRAM"` 或 `"SSD"` | [src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h)，决定索引结构的存储介质 |
| `value_type` | string | 是 | 值存储类型：`"DRAM"`、`"SSD"` 或 `"HYBRID"` | [src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h)，决定数据值的存储介质 |
| `capacity` | integer | 是 | 预估存储条目数 | [src/storage/kv_engine/base_kv.h](../src/storage/kv_engine/base_kv.h)，用于空间预分配和性能优化 |
| `value_size` | integer | 是 | 单条值的字节数 | [src/storage/kv_engine/base_kv.h](../src/storage/kv_engine/base_kv.h)，定义每个键对应的值大小 |

#### HYBRID 模式必填字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `shmcapacity` | integer | 是 | DRAM/共享内存侧的字节数容量 |
| `ssdcapacity` | integer | 是 | SSD 侧的字节数容量 |

注：HYBRID 模式不需要 `capacity` 和 `value_size`。

#### 可选字段

| 字段 | 类型 | 默认值 | 说明 | 作用位置 |
|------|------|--------|------|----------|
| `value_memory_management` | string | `"PersistLoopShmMalloc"` | 内存管理器类型：`"PersistLoopShmMalloc"` 或 `"R2ShmMalloc"` | [src/storage/hybrid/value.h](../src/storage/hybrid/value.h)，控制底层内存分配策略 |

#### 引擎类型自动推导规则

`base::ResolveEngine` 根据 `index_type` 和 `value_type` 组合自动推导引擎类型：

- `value_type = "HYBRID"` → `KVEngineHybrid`
- `value_type = "DRAM"` 或 `"SSD"`：
  - `index_type = "DRAM"` → `KVEngineExtendibleHash`
  - `index_type = "SSD"` → `KVEngineCCEH`

引擎推导实现位置：[src/storage/kv_engine/engine_selector.h](../src/storage/kv_engine/engine_selector.h)

## 2. distributed_client 配置

`distributed_client` 配置用于分布式客户端，实现多分片参数服务器的访问。

| 字段 | 类型 | 必填 | 说明 | 作用位置 |
|------|------|------|------|----------|
| `num_shards` | integer | 是 | 分片总数，应与 `cache_ps.num_shards` 一致 | [src/grpc_ps/dist_brpc_ps_client.cpp](../src/grpc_ps/dist_brpc_ps_client.cpp)、[src/grpc_ps/dist_grpc_ps_client.cpp](../src/grpc_ps/dist_grpc_ps_client.cpp)，用于数据分片路由 |
| `hash_method` | string | 是 | 哈希方法：`"city_hash"` 或 `"simple_mod"` | [src/grpc_ps/dist_brpc_ps_client.cpp](../src/grpc_ps/dist_brpc_ps_client.cpp)，`GetShardId` 函数根据此字段选择哈希算法，`city_hash` 使用 CityHash 算法，`simple_mod` 使用简单取模 |
| `max_keys_per_request` | integer | 否（默认 500） | 单次请求最大键数量 | [src/grpc_ps/dist_brpc_ps_client.cpp](../src/grpc_ps/dist_brpc_ps_client.cpp)，`PartitionKeys` 函数用此字段限制单个分片请求的键数量，防止请求过大 |
| `servers` | array | 是 | 服务器节点配置，结构同 `cache_ps.servers` | [src/grpc_ps/dist_brpc_ps_client.cpp](../src/grpc_ps/dist_brpc_ps_client.cpp)，`InitializeClients` 函数遍历此数组创建到各分片的客户端连接 |

## 3. client 配置

`client` 配置用于单节点客户端，直接连接单个参数服务器。

| 字段 | 类型 | 必填 | 说明 | 作用位置 |
|------|------|------|------|----------|
| `host` | string | 是 | 服务器主机地址 | [src/grpc_ps/grpc_ps_client.cpp](../src/grpc_ps/grpc_ps_client.cpp)，`GRPCParameterClient` 构造函数使用此字段建立 gRPC 通道 |
| `port` | integer | 是 | 服务器端口 | [src/grpc_ps/grpc_ps_client.cpp](../src/grpc_ps/grpc_ps_client.cpp)，与 `host` 组合形成连接地址 |
| `shard` | integer | 是 | 目标分片编号 | [src/grpc_ps/grpc_ps_client.h](../src/grpc_ps/grpc_ps_client.h)，用于标识连接的分片 |

## 配置示例

### 完整配置示例

```json
{
    "cache_ps": {
        "ps_type": "GRPC",
        "max_batch_keys_size": 65536,
        "num_threads": 32,
        "num_shards": 2,
        "servers": [
            {
                "host": "127.0.0.1",
                "port": 15000,
                "shard": 0
            },
            {
                "host": "127.0.0.1",
                "port": 15001,
                "shard": 1
            }
        ],
        "base_kv_config": {
            "path": "/tmp/recstore_data",
            "capacity": 40000000,
            "value_size": 512,
            "value_type": "DRAM",
            "index_type": "DRAM",
            "value_memory_management": "PersistLoopShmMalloc"
        }
    },
    "distributed_client": {
        "num_shards": 2,
        "hash_method": "city_hash",
        "max_keys_per_request": 500,
        "servers": [
            {
                "host": "127.0.0.1",
                "port": 15000,
                "shard": 0
            },
            {
                "host": "127.0.0.1",
                "port": 15001,
                "shard": 1
            }
        ]
    },
    "client": {
        "host": "127.0.0.1",
        "port": 15000,
        "shard": 0
    }
}
```

### DRAM 索引 + SSD 值配置

```json
{
    "cache_ps": {
        "base_kv_config": {
            "path": "/data/recstore",
            "index_type": "DRAM",
            "value_type": "SSD",
            "capacity": 2000000,
            "value_size": 128,
            "value_memory_management": "PersistLoopShmMalloc"
        }
    }
}
```

推导引擎类型：`KVEngineExtendibleHash`

### SSD 索引 + SSD 值配置

```json
{
    "cache_ps": {
        "base_kv_config": {
            "path": "/data/recstore",
            "index_type": "SSD",
            "value_type": "SSD",
            "capacity": 2000000,
            "value_size": 128,
            "value_memory_management": "PersistLoopShmMalloc"
        }
    }
}
```

推导引擎类型：`KVEngineCCEH`

### HYBRID 混合模式配置

```json
{
    "cache_ps": {
        "base_kv_config": {
            "path": "/data/recstore",
            "index_type": "DRAM",
            "value_type": "HYBRID",
            "shmcapacity": 10737418240,
            "ssdcapacity": 107374182400,
            "value_memory_management": "PersistLoopShmMalloc"
        }
    }
}
```

推导引擎类型：`KVEngineHybrid`

### 在 CI 中的配置

Github Actions 服务器环境限制 CPU 使用，同时无显卡支持，在 ci 配置脚本中使用：

```shell
jq '.cache_ps.base_kv_config.capacity = 512
    | .cache_ps.max_batch_keys_size = 128
    | .cache_ps.num_threads = 4
    | .distributed_client.max_keys_per_request = 32
    | .cache_ps.base_kv_config.index_type = "DRAM"
    | .cache_ps.base_kv_config.value_type = "SSD"
    | .cache_ps.base_kv_config.type = "DRAM"
    | .cache_ps.base_kv_config.queue_size = 1024'
```

来配置DRAM 索引 + SSD 值配置。

## 配置文件使用

配置文件通常保存为 `recstore_config.json`，可通过以下方式读取：

```cpp
std::ifstream config_file("recstore_config.json");
nlohmann::json config;
config_file >> config;

// 服务器端使用
auto cache_ps = std::make_unique<CachePS>(config["cache_ps"]);

// 分布式客户端使用
auto dist_client = std::make_unique<DistributedBRPCParameterClient>(config);

// 单节点客户端使用
auto client = std::make_unique<GRPCParameterClient>(config["client"]);
```

## 注意事项

1. `cache_ps.num_shards` 必须与 `cache_ps.servers` 数组长度一致
2. `distributed_client.num_shards` 应与 `cache_ps.num_shards` 保持一致
3. `distributed_client.servers` 配置应与 `cache_ps.servers` 保持一致
4. `base_kv_config.path` 建议为每个实例分配独立目录，避免冲突
5. HYBRID 模式需要提供 `shmcapacity` 和 `ssdcapacity`，不需要 `capacity` 和 `value_size`
6. 非 HYBRID 模式必须提供 `capacity` 和 `value_size`
7. `hash_method` 推荐使用 `city_hash`，性能优于 `simple_mod`
8. `value_memory_management` 默认使用 `PersistLoopShmMalloc`，特殊场景可选择 `R2ShmMalloc`
