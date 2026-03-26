# EmbeddingBag / DistEmbedding 模块

## 概述

PyTorch 提供两种分布式嵌入模块：

1. **RecStoreEmbeddingBagCollection** - 用于 torchrec 风格的模型，支持多特征融合预取
2. **DistEmbedding** - 用于单个嵌入表，支持自定义优化器

两者均位于 `src/python/pytorch/` 目录。

## RecStoreEmbeddingBagCollection

多特征嵌入包模块，适合 DLRM、Wide&Deep 等推荐系统模型。

### 初始化

```python
emb_bag = RecStoreEmbeddingBagCollection(
    embedding_bag_configs=[
        {
            "name": "user_emb",
            "num_embeddings": 100000,
            "embedding_dim": 64,
            "feature_names": ["user_id"],
        },
        {
            "name": "item_emb",
            "num_embeddings": 50000,
            "embedding_dim": 64,
            "feature_names": ["item_id", "category_id"],
        },
    ],
    lr=0.01,
    enable_fusion=True,
    fusion_k=30,
    ps_host=None,
    ps_port=None
)
```

**配置参数**

| 参数 | 说明 |
|------|------|
| embedding_bag_configs | EmbeddingBagConfig 列表，每个定义一个嵌入表 |
| lr | 优化器学习率 |
| enable_fusion | 是否启用融合预取 (默认 True) |
| fusion_k | 融合 ID 的位移量 (默认 30) |
| ps_host | (可选) 远程 PS 服务器地址 |
| ps_port | (可选) 远程 PS 服务器端口 |

**EmbeddingBagConfig**

| 字段 | 说明 |
|------|------|
| name | 嵌入表名称 |
| num_embeddings | 嵌入数量 |
| embedding_dim | 嵌入维度 |
| feature_names | 该表关联的特征列表 |

### 前向传播

**工作流程**

| 步骤 | 代码/操作 | 说明 |
|------|----------|------|
| 1 | 接收 `features: KeyedJaggedTensor` | 稀疏特征输入 |
| 2 | `config = module.embedding_bag_configs()[0]` | 获取嵌入表配置 |
| 3 | `all_embeddings = module.kv_client.pull(name, ids)` | 批量读取嵌入向量 |
| 4 | `all_embeddings.requires_grad = True` | 启用梯度计算 |
| 5 | `F.embedding_bag(input, weight, offsets, mode="sum")` | pooling 聚合 |
| 6 | 返回 `[batch_size, num_features, emb_dim]` | 最终输出 |

**融合 ID 计算**

启用融合时，全局 ID 计算：
```
global_id = original_id + (table_idx << fusion_k)
```

用于合并请求。

### 反向传播

自定义 autograd 函数记录梯度

**梯度处理流程**

| 步骤 | 代码/操作 | 说明 |
|------|----------|------|
| 1 | 接收 `grad_output` | 上层梯度 |
| 2 | `for i, key in enumerate(feature_keys)` | 遍历每个特征 |
| 3 | `offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths_cpu, 0)])` | 计算 bag 边界 |
| 4 | `ids_to_update = values_cpu[start:end]` | 提取该 bag 的 ID |
| 5 | `grad_for_bag = grad_output_reshaped[sample_idx, i]` | 提取该 bag 的梯度 |
| 6 | `module._trace.append((config_name, ids, grads))` | 追踪梯度 |
| 7 | `return None, None, None, None` | 不计算输入梯度 |

**梯度追踪**

```python
self._trace.append(
    (config_name, ids_to_update.detach(), grads_to_trace.detach())
)
```

外部优化器需处理 `_trace` 列表：

```python
for table_name, ids, grads in module._trace:
    client.update(table_name, ids, grads)
module.reset_trace()
```

### 异步预取

#### 发起预取

```python
def issue_fused_prefetch(
    features: KeyedJaggedTensor,
    record_handle: bool = True
) -> int | Tuple:
```

计算融合 ID 并异步预取

**工作流程**（`src/python/pytorch/torchrec/EmbeddingBag.py:issue_fused_prefetch`）

| 步骤 | 代码/操作 | 说明 |
|------|----------|------|
| 1 | `for key in keys_in_batch` | 遍历批次中的特征 |
| 2 | `values = kjt_per_feature.values()` | 获取特征值 |
| 3 | `prefix = (table_idx << fusion_k)` | 计算融合前缀 |
| 4 | `fused_values = values + prefix` | 融合成全局 ID |
| 5 | `unique_ids, inverse = torch.unique(fused_values_all, return_inverse=True)` | ID 去重 |
| 6 | `handle = self.kv_client.prefetch(unique_ids)` | 异步预取 |
| 7 | 返回 `(handle, num_ids, issue_ts, ...)` 或仅 handle | 返回预取信息 |

**返回值**

- record_handle=True: 返回 prefetch_id
- record_handle=False: 返回 (prefetch_id, num_ids, issue_ts, fused_ids, inverse)

#### 设置预取句柄

```python
def set_fused_prefetch_handle(
    handle: int,
    num_ids: int | None = None,
    issue_ts: float | None = None,
    record_stats: bool = True,
    fused_ids_cpu: torch.Tensor | None = None,
    fused_inverse: torch.Tensor | None = None
):
```

手动设置预取句柄（通常由生产者线程调用）

#### 等待预取

前向传播时，如果已设置 _fused_prefetch_handle，自动等待其完成

### 成员变量

| 成员 | 类型 | 说明 |
|------|------|------|
| _embedding_bag_configs | List[EmbeddingBagConfig] | 嵌入表配置 |
| kv_client | RecStoreClient | KV 客户端 |
| feature_keys | List[str] | 所有特征名 |
| _config_names | Dict | 特征→表名映射 |
| _trace | List | 梯度追踪列表 |
| _prefetch_handles | Dict | 特征→预取 ID 映射 |
| _fused_prefetch_handle | int | 融合预取 ID |

## DistEmbedding

单个嵌入表模块，支持自定义梯度处理

### 初始化

```python
emb = DistEmbedding(
    num_embeddings=100000,
    embedding_dim=64,
    name="user_embedding",
    init_func=lambda shape, dtype: torch.randn(shape, dtype=dtype) * 0.01,
    lr=0.01
)
```

**参数**

| 参数 | 说明 |
|------|------|
| num_embeddings | 嵌入数量 |
| embedding_dim | 嵌入维度 |
| name | 表名称 (必须唯一) |
| init_func | 初始化函数 (可选) |
| lr | 学习率 |

### 前向传播

**工作流程**

| 步骤 | 代码位置 | 代码/操作 | 说明 |
|------|---------|----------|------|
| 1 | `src/python/pytorch/recstore/DistEmb.py:DistEmbedding.forward` | `ids` 输入 `[N]` | 嵌入 ID |
| 2 | 同上 | `_DistEmbFunction.apply(ids, ...)` | 调用自定义 autograd function |
| 3 | `_DistEmbFunction.forward` | `embs = dist_tensor[ids]` | 通过 DistTensor 执行查询 |
| 4 | `DistTensor.__getitem__` | `client.pull(name, ids)` | 从 KVClient 拉取向量 |
| 5 | `RecStoreClient.pull` | `torch.ops.emb_read(ids, embedding_dim)` | 调用 C++ 扩展读取 |
| 6 | `_DistEmbFunction.forward` | 返回 `embs` | 返回嵌入向量 `[N, D]` |

### 反向传播

自定义 Function 的 backward

**梯度处理**（`src/python/pytorch/recstore/DistEmb.py:_DistEmbFunction.backward`）

| 步骤 | 代码/操作 | 说明 |
|------|----------|------|
| 1 | `ids, = ctx.saved_tensors` | 恢复保存的 ID |
| 2 | `module_instance = ctx.module_instance` | 获取模块实例 |
| 3 | `unique_ids, inverse = torch.unique(ids_cpu, return_inverse=True)` | ID 去重并保留逆映射 |
| 4 | `grad_sum = torch.zeros((unique_ids.size(0), grad_cpu.size(1)))` | 创建聚合梯度 |
| 5 | `grad_sum.index_add_(0, inverse, grad_cpu)` | 相同 ID 梯度累加 |
| 6 | `module_instance._trace.append((unique_ids, grad_sum))` | 追踪聚合梯度 |
| 7 | `return None, None, None, None` | 返回（不计算输入梯度） |

### DistTensor

底层分布式张量抽象

```python
def __init__(
    self,
    shape: Tuple,
    dtype: torch.dtype,
    name: str,
    init_func: Optional[Callable] = None,
    persistent: bool = False,
    is_gdata: bool = True
):
```

**特点**

- 单个全局张量视图
- 支持 __getitem__ (查询) 和 __setitem__ (更新)
- 自动初始化与清理

**操作**

| 操作 | 说明 |
|------|------|
| tensor[ids] | 等价于 client.pull(name, ids) |
| tensor[ids] = values | 等价于 client.push(name, ids, values) |