# CommonOp 接口

## 概述

`CommonOp` 是 C++ 侧的统一接口，定义了所有嵌入操作的规范。位于 `src/framework/op.h`。

## 核心接口

### 初始化

```cpp
virtual void EmbInit(const RecTensor& keys, const RecTensor& init_values) = 0;
virtual void EmbInit(const RecTensor& keys, const InitStrategy& strategy) = 0;
virtual bool InitEmbeddingTable(const std::string& table_name, 
                                const EmbeddingTableConfig& config) = 0;
virtual void SetPSConfig(const std::string& host, int port) = 0;
```

| 方法 | 说明 |
|------|------|
| EmbInit(keys, values) | 用提供的值初始化嵌入 |
| EmbInit(keys, strategy) | 根据策略 (Normal/Uniform/Xavier/Zero) 初始化 |
| InitEmbeddingTable | 创建新的嵌入表 |
| SetPSConfig | 动态设置 PS 连接配置 (Host/Port) |

**InitStrategy 支持**

| 类型 | 参数 |
|------|------|
| Normal | mean, std |
| Uniform | lower, upper |
| Xavier | 自动计算 |
| Zero | 全零 |

### 同步读写

```cpp
virtual void EmbRead(const RecTensor& keys, RecTensor& values) = 0;
virtual void EmbWrite(const RecTensor& keys, const RecTensor& values) = 0;
```

| 方法 | 说明 | 使用场景 |
|------|------|---------|
| EmbRead | 批量读取嵌入向量 | 前向传播 |
| EmbWrite | 批量写入嵌入向量 | 初始化、检查点加载 |

**参数**

| 参数    | 类型           | 形状      | 说明         |
|---------|----------------|-----------|--------------|
| keys    | uint64 张量    | [N]       | 嵌入 ID      |
| values  | float32 张量   | [N, D]    | 嵌入向量     |

### 异步预取

```cpp
virtual uint64_t EmbPrefetch(const RecTensor& keys, const RecTensor& values) = 0;
virtual bool IsPrefetchDone(uint64_t prefetch_id) = 0;
virtual void WaitForPrefetch(uint64_t prefetch_id) = 0;
virtual void GetPretchResult(uint64_t prefetch_id, 
                             std::vector<std::vector<float>>* values) = 0;
```

| 方法 | 说明 |
|------|------|
| EmbPrefetch | 异步读取，返回唯一 ID |
| IsPrefetchDone | 检查预取是否完成 |
| WaitForPrefetch | 阻塞等待预取完成 |
| GetPretchResult | 获取预取结果 |

**工作流程**

| 步骤 | 代码/操作 | 说明 |
|------|----------|------|
| 1 | `pid = EmbPrefetch(keys, dummy_values)` | 发起异步预取 |
| 2 | 后台执行 | 不阻塞当前线程 |
| 3 | `while not IsPrefetchDone(pid): continue` | 轮询检查状态 |
| 4 | `WaitForPrefetch(pid)` | 阻塞等待完成 |
| 5 | `GetPretchResult(pid, &results)` | 获取预取结果 |
| 6 | 返回 `vector<vector<float>>` | 返回向量数据 |

### 异步写入

```cpp
virtual uint64_t EmbWriteAsync(const RecTensor& keys, 
                               const RecTensor& values) = 0;
virtual bool IsWriteDone(uint64_t write_id) = 0;
virtual void WaitForWrite(uint64_t write_id) = 0;
```

| 方法 | 说明 |
|------|------|
| EmbWriteAsync | 异步写入，返回写入 ID **(未实现)** |
| IsWriteDone | 检查写入是否完成 **(未实现)** |
| WaitForWrite | 阻塞等待写入完成 **(未实现)** |

### 梯度更新

```cpp
virtual void EmbUpdate(const RecTensor& keys, const RecTensor& grads) = 0;
virtual void EmbUpdate(const std::string& table_name, 
                       const RecTensor& keys, 
                       const RecTensor& grads) = 0;
```

| 方法 | 说明 |
|------|------|
| EmbUpdate(keys, grads) | 对所有表的指定 ID 更新梯度 |
| EmbUpdate(table_name, keys, grads) | 对特定表的指定 ID 更新梯度 |

**梯度更新流程**

| 步骤 | 代码位置 | 说明 |
|------|---------|------|
| 1 | `src/framework/op.h:CommonOp` | 调用 EmbUpdate 将 (ID, grad) 送往 PS |
| 2 | `src/ps/client/` | PS 客户端与服务器通信 |
| 3 | `src/ps/server/` | PS 内部执行 SGD: emb -= lr * grad |
| 4 | `src/storage/` | 更新后的嵌入向量存入存储层 |
| 5 | 后续读取 | 自动返回最新的嵌入向量 |

### 存在性检查与删除

```cpp
virtual bool EmbExists(const RecTensor& keys) = 0;
virtual void EmbDelete(const RecTensor& keys) = 0;
```

| 方法 | 说明 |
|------|------|
| EmbExists | 检查 ID 是否存在 |
| EmbDelete | 删除指定 ID 的嵌入 |

### 持久化

```cpp
virtual void SaveToFile(const std::string& path) = 0;
virtual void LoadFromFile(const std::string& path) = 0;
```

| 方法 | 说明 |
|------|------|
| SaveToFile | 将嵌入检查点保存到文件 |
| LoadFromFile | 从文件加载嵌入 |

## KVClientOp 实现

`KVClientOp` 是 `CommonOp` 的标准实现，通过 `BasePSClient` 与参数服务器通信。

**成员变量**

| 成员 | 类型 | 说明 |
|------|------|------|
| embedding_dim_ | int64_t | 嵌入向量维度 |
| ps_client_ | BasePSClient* | 参数服务器客户端 |

**创建实例**

```cpp
std::shared_ptr<CommonOp> op = GetKVClientOp();
```

## 数据类型

### RecTensor

用于跨 C++ 和 Python 传递数据的张量包装：

| 字段 | 说明 |
|------|------|
| data_ptr | 数据缓冲区指针 |
| shape | 张量形状 |
| dtype | 数据类型 (UINT64, FLOAT32) |

### InitStrategy

| 字段 | 说明 |
|------|------|
| type | InitStrategyType 枚举 |
| mean, std | Normal 初始化参数 |
| lower, upper | Uniform 初始化参数 |

### EmbeddingTableConfig

定义嵌入表的配置（详见参数服务器文档）。

## 线程安全

`KVClientOp` 可在多线程环境下安全使用：

- 每个线程可独立调用接口
- 参数服务器客户端内部处理线程同步
