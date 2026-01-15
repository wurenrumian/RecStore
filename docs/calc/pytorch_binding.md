# PyTorch C++ 扩展

## 概述

PyTorch C++ 扩展位于 `src/framework/pytorch/op_torch.cc`，将 CommonOp 接口暴露为 torch.ops 操作，支持 CPU/GPU 张量。

## 操作列表

### 嵌入读取

```cpp
torch.ops.recstore_ops.emb_read(keys, embedding_dim) → Tensor
```

**函数签名**
```cpp
torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim)
```

| 参数 | 说明 |
|------|------|
| keys | int64 张量，[N] 形状，可在 CPU/GPU |
| embedding_dim | 整数，嵌入维度 |

| 返回值 | 说明 |
|--------|------|
| values | float32 张量，[N, embedding_dim] 形状，CPU 上 |

**工作流程**

| 步骤 | 代码位置 | 代码/操作 | 说明 |
|------|---------|----------|------|
| 1 | `src/framework/pytorch/op_torch.cc:emb_read_torch` | 验证 keys 张量 (int64, 1D, contiguous) | 输入检查 |
| 2 | 同上 | `if (keys.is_cuda()) cpu_keys = keys.cpu()` | GPU → CPU 复制 |
| 3 | 同上 | `base::RecTensor rec_keys = ToRecTensor(cpu_keys, UINT64)` | 转换为 RecTensor |
| 4 | 同上 | `op->EmbRead(rec_keys, rec_values)` | 调用 C++ op 读取 |
| 5 | 同上 | `if (values.is_cuda()) values.copy_(cpu_values)` | CPU → GPU 复制 |
| 6 | 同上 | 返回 `values` | 返回 float32 张量 `[N, embedding_dim]` |

**示例**
```python
import torch
from recstore.KVClient import get_kv_client

client = get_kv_client()
keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
embeddings = torch.ops.recstore_ops.emb_read(keys, 128)  # [5, 128]
```

### 嵌入写入

```cpp
torch.ops.recstore_ops.emb_write(keys, values)
```

**函数签名**
```cpp
void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values)
```

| 参数 | 说明 |
|------|------|
| keys | int64 张量，[N] |
| values | float32 张量，[N, D] |

调用 op->EmbWrite() 同步写入嵌入

### 异步预取

```cpp
torch.ops.recstore_ops.emb_prefetch(keys) → int64
```

**函数签名**
```cpp
int64_t emb_prefetch_torch(const torch::Tensor& keys)
```

返回预取 ID (uint64_t 转换为 int64_t)

**工作流程** （`src/framework/pytorch/op_torch.cc:emb_prefetch_torch`）

| 步骤 | 代码/操作 | 说明 |
|------|----------|------|
| 1 | 验证 keys 张量 | 输入检查 |
| 2 | `if (keys.is_cuda()) cpu_keys = keys.cpu()` | GPU → CPU |
| 3 | `base::RecTensor rec_keys = ToRecTensor(cpu_keys, UINT64)` | 转换为 RecTensor |
| 4 | `uint64_t pid = op->EmbPrefetch(rec_keys, rec_vals)` | 发起异步预取 |
| 5 | 返回 `static_cast<int64_t>(pid)` | 返回预取 ID |

### 等待预取结果

```cpp
torch.ops.recstore_ops.emb_wait_result(prefetch_id, embedding_dim) → Tensor
```

**函数签名**
```cpp
torch::Tensor emb_wait_result_torch(int64_t prefetch_id, int64_t embedding_dim)
```

| 参数 | 说明 |
|------|------|
| prefetch_id | int64，从 prefetch 返回的 ID |
| embedding_dim | 整数，嵌入维度 |

| 返回值 | 说明 |
|--------|------|
| values | float32 张量，[N, embedding_dim] |

**工作流程**
1. 调用 op->WaitForPrefetch()
2. 调用 op->GetPretchResult()
3. 将 vector<vector<float>> 转换为 Tensor
4. 返回 [N, D] 张量

### 梯度更新

```cpp
torch.ops.recstore_ops.emb_update(keys, grads)
torch.ops.recstore_ops.emb_update_with_table(table_name, keys, grads)
```

**函数签名**
```cpp
void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads)
void emb_update_with_table_torch(
    const std::string& table_name,
    const torch::Tensor& keys,
    const torch::Tensor& grads
)
```

| 参数 | 说明 |
|------|------|
| keys | int64 张量，[N] |
| grads | float32 张量，[N, D] |
| table_name | 嵌入表名称 |

调用 op->EmbUpdate() 应用梯度

### 初始化嵌入表

```cpp
torch.ops.recstore_ops.init_embedding_table(name, num_embeddings, embedding_dim) → bool
```

**函数签名**
```cpp
bool init_embedding_table_torch(
    const std::string& name,
    int64_t num_embeddings,
    int64_t embedding_dim
)
```

| 参数 | 说明 |
|------|------|
| name | 表名称 |
| num_embeddings | 嵌入数量 |
| embedding_dim | 嵌入维度 |

| 返回值 | 说明 |
|--------|------|
| success | bool，初始化是否成功 |

## RecTensor 转换

### ToRecTensor 函数

```cpp
static inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype)
```

将 PyTorch Tensor 转换为 RecTensor 用于 C++ 侧处理

| 操作 | 说明 |
|------|------|
| 提取数据指针 | `tensor.data_ptr()` |
| 提取形状 | 遍历 `tensor.dim()` 获取各维大小 |
| 指定数据类型 | UINT64 (keys) 或 FLOAT32 (values) |

## 日志系统

使用环境变量控制日志级别：

```bash
export RECSTORE_LOG_LEVEL=3  # DEBUG
export RECSTORE_LOG_LEVEL=2  # INFO (默认)
export RECSTORE_LOG_LEVEL=1  # WARNING
export RECSTORE_LOG_LEVEL=0  # ERROR
```

**日志宏**
```cpp
RECSTORE_LOG(level, message)
```

输出示例：
```
[INFO] emb_read_torch: keys shape=[1000], dtype=Int64
[DEBUG] emb_read_torch: calling op->EmbRead
```

## 设备支持

| 设备 | 说明 |
|------|------|
| CPU | 直接处理 |
| GPU (CUDA) | 自动复制到 CPU，处理后复制回 GPU |

**转移逻辑**

| 步骤 | 操作 | 代码位置 |
|------|------|---------|
| 1 | GPU Tensor 复制到 CPU | `src/framework/pytorch/op_torch.cc` |
| 2 | C++ 处理 (op->EmbRead 等) | `src/framework/op.h:KVClientOp` |
| 3 | CPU Tensor 复制回 GPU | `src/framework/pytorch/op_torch.cc` |

## 张量验证

操作前进行的检查：

| 检查项 | 说明 |
|--------|------|
| 维度 | keys 必须是 1D，values 必须是 2D |
| 数据类型 | keys 必须是 int64，values 必须是 float32 |
| 连续性 | 张量必须是 contiguous |
| 大小 | embedding_dim > 0，keys.size() > 0 |

失败时抛出 `TORCH_CHECK` 异常

## 编译

编译命令：
```bash
cd src/framework/pytorch
g++ -c op_torch.cc -I/path/to/torch/include -I/path/to/recstore/include
```

生成 `lib_recstore_ops.so`，可由 Python 通过 `torch.ops.load_library()` 加载