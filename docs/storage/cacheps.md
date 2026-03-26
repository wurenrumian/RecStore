# CachePS 层

## 概述

CachePS 是参数服务器与存储引擎之间的接口层，位于 `ps/base/cache_ps_impl.h`，负责将参数服务器的参数操作转换为 BaseKV 的 KV 存储操作。

## 类定义

```cpp
class CachePS {
  using key_t = uint64_t;
  
  CachePS(json config);
  bool Initialize(model_config_path, emb_file_path);
  void PutParameter(reader, tid);
  void GetParameter(keys, output);
  void Clear();
};
```

## 核心功能

### 初始化

CachePS 根据配置创建 BaseKV 实例：

| 步骤 | 操作 |
|------|------|
| 1 | 解析 JSON 配置 |
| 2 | 构造 BaseKVConfig |
| 3 | 调用 ResolveEngine 推导引擎类型 |
| 4 | 使用工厂创建 BaseKV 实例 |

配置结构：

```json
{
  "num_threads": 16,
  "base_kv_config": {
    "path": "/data/recstore",
    "index_type": "DRAM",
    "value_type": "SSD",
    "capacity": 1000000,
    "value_size": 128
  }
}
```

### 参数操作

**PutParameter** - 批量写入参数

```cpp
void PutParameter(
  coroutine<void>::push_type& sink,
  const ParameterCompressReader* reader,
  int tid
)
```

工作流程：
1. 从 reader 解压参数数据
2. 提取 keys 和 values
3. 调用 BaseKV.BatchPut
4. 通过协程支持大批量数据

**GetParameter** - 批量读取参数

```cpp
void GetParameter(
  const base::ConstArray<key_t>& keys,
  base::MutableArray<ParameterPack>& packs,
  int tid
)
```

工作流程：
1. 准备输出缓冲区
2. 调用 BaseKV.BatchGet
3. 将结果打包到 ParameterPack

**PutSingleParameter** - 单个参数写入

```cpp
void PutSingleParameter(
  const uint64_t key, 
  const void* data, 
  const int dim, 
  const int tid
)
```

直接调用 BaseKV.Put，适用于单个参数的更新。

## 数据结构

### TaskElement

用于异步任务队列的元素：

| 字段 | 类型 | 说明 |
|------|------|------|
| keys | ConstArray<key_t> | 参数键数组 |
| packs | MutableArray<ParameterPack> | 输出打包数组 |
| promise | atomic_bool* | 完成标志 |

### ParameterPack

参数压缩打包格式：

| 字段 | 说明 |
|------|------|
| key | 参数键 |
| dim | 参数维度 |
| emb_data | 浮点数据指针 |

## 与 BaseKV 的交互

CachePS 通过以下接口与 BaseKV 交互：

| CachePS 方法 | BaseKV 方法 | 说明 |
|-------------|-------------|------|
| PutParameter | BatchPut | 批量写入 |
| GetParameter | BatchGet | 批量读取 |
| PutSingleParameter | Put | 单个写入 |

## 成员变量

| 成员 | 类型 | 说明 |
|------|------|------|
| base_kv_ | unique_ptr<BaseKV> | KV 引擎实例 |
