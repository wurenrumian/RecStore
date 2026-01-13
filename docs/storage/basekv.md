# BaseKV 抽象层

## 概述

BaseKV 是所有 KV 引擎的抽象基类，定义了统一的存储接口。位于 `storage/kv_engine/base_kv.h`。

## 核心接口

```cpp
class BaseKV {
  virtual void Get(uint64_t key, string& value, unsigned tid) = 0;
  virtual void Put(uint64_t key, string_view& value, unsigned tid) = 0;
  virtual void BatchGet(ConstArray<uint64_t> keys, 
                        vector<ConstArray<float>>* values,
                        unsigned tid) = 0;
  virtual void BatchPut(coroutine<void>::push_type& sink,
                        ConstArray<uint64_t> keys,
                        vector<ConstArray<float>>* values,
                        unsigned tid);
};
```

## 方法说明

| 方法 | 参数 | 说明 |
|------|------|------|
| Get | key, value, tid | 读取单个键值对 |
| Put | key, value_view, tid | 写入单个键值对 |
| BatchGet | keys, values, tid | 批量读取 |
| BatchPut | sink, keys, values, tid | 批量写入（协程） |

参数说明：
- `tid` - 线程 ID，用于线程安全的内存管理
- `sink` - 协程 push_type，支持大批量数据的分批处理
- `ConstArray` - 只读数组视图，避免数据拷贝
- `MutableArray` - 可写数组视图

## 配置结构

### BaseKVConfig

```cpp
struct BaseKVConfig {
  int num_threads_;        // 线程数
  json json_config_;       // JSON 配置
};
```

### 配置字段

**必填字段（非 HYBRID）**

| 字段 | 类型 | 说明 |
|------|------|------|
| path | string | 工作目录 |
| index_type | string | "DRAM" 或 "SSD" |
| value_type | string | "DRAM" 或 "SSD" |
| capacity | uint64_t | 预估条目数 |
| value_size | int | 每个值的字节数 |

**必填字段（HYBRID）**

| 字段 | 类型 | 说明 |
|------|------|------|
| path | string | 工作目录 |
| index_type | string | "DRAM" 或 "SSD" |
| value_type | string | "HYBRID" |
| shmcapacity | uint64_t | DRAM 字节数 |
| ssdcapacity | uint64_t | SSD 字节数 |

**可选字段**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| value_memory_management | "PersistLoopShmMalloc" | 内存管理器类型 |

## 引擎选择机制

通过 `base::ResolveEngine(config)` 根据配置自动选择引擎：

```cpp
auto r = base::ResolveEngine(kv_config);
// r.engine - 引擎类型字符串
// r.cfg - 补全后的配置
```

引擎选择规则：

| index_type | value_type | 选择的引擎 |
|------------|------------|-----------|
| DRAM | SSD | KVEngineExtendibleHash |
| SSD | SSD | KVEngineCCEH |
| DRAM/SSD | HYBRID | KVEngineHybrid |

## 工厂创建

使用工厂模式创建引擎实例：

```cpp
std::unique_ptr<BaseKV> kv(
  base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
    engine_type, config
  )
);
```

## 线程安全

BaseKV 通过 `tid` 参数实现线程安全：
- 每个线程有独立的 tid
- 内存管理器根据 tid 使用独立的数据结构
- 避免锁竞争，提高并发性能

## 依赖

使用 BaseKV 前需要：
1. `#include "storage/kv_engine/engine_factory.h"` - 注册所有引擎
2. `#include "memory/memory_factory.h"` - 注册所有内存管理器
