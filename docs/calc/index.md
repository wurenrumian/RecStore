# RecStore 计算层

## 概述

计算层负责模型训练和推理中的嵌入向量查询与梯度更新，从参数服务器向上，通过 OP 层提供统一的 C++ 接口，再到 PyTorch Python 绑定，最后接入推荐系统模型代码。

## 架构分层

| 层级 | 模块 | 文件路径 | 说明 |
|------|------|---------|------|
| 7 | 推荐模型 | `用户代码` | DLRM, Wide&Deep 等 |
| 6 | Embedding 模块 | `src/python/pytorch/torchrec/EmbeddingBag.py` | EmbeddingBag / DistEmbedding |
| 5 | KV 客户端 | `src/python/pytorch/recstore/KVClient.py` | RecStoreClient 单例 |
| 4 | PyTorch 扩展 | `src/framework/pytorch/op_torch.cc` | torch.ops.recstore_ops |
| 3 | C++ 接口 | `src/framework/op.h` | CommonOp / KVClientOp |
| 2 | PS 客户端 | `src/ps/client/` | BasePSClient |
| 1 | PS 服务 | `src/ps/server/` | gRPC/bRPC 通信 |
| 0 | 存储层 | `src/storage/` | BaseKV / 引擎 / 内存管理 |

## 数据流

### 前向传播

| 步骤 | 组件 | 文件路径 | 代码/操作 |
|------|------|---------|----------|
| 1 | 推荐模型 | 用户代码 | `features = get_batch()` |
| 2 | EmbeddingBag | `src/python/pytorch/torchrec/EmbeddingBag.py` | `output = emb_module(features)` |
| 3 | KVClient | `src/python/pytorch/recstore/KVClient.py` | `client.pull(name, ids)` |
| 4 | PyTorch 扩展 | `src/framework/pytorch/op_torch.cc` | `torch.ops.recstore_ops.emb_read()` |
| 5 | CommonOp | `src/framework/op.h` | `op->EmbRead(rec_keys, rec_values)` |
| 6 | BasePSClient | `src/ps/client/` | 与 PS 通信获取向量 |
| 7 | EmbeddingBag | `src/python/pytorch/torchrec/EmbeddingBag.py` | `F.embedding_bag(..., mode="sum")` |
| 8 | 返回 | PyTorch | `[batch_size, num_features, emb_dim]` |

### 反向传播与梯度更新

| 步骤 | 组件 | 文件路径 | 代码/操作 |
|------|------|---------|----------|
| 1 | 用户代码 | - | `loss.backward()` |
| 2 | EmbeddingBag | `src/python/pytorch/torchrec/EmbeddingBag.py` | 触发 _RecStoreEBCFunction.backward |
| 3 | 梯度收集 | 同上 | 将 (ID, grad) 追踪到 `_trace` |
| 4 | 优化器 | `src/python/pytorch/recstore/optimizer.py` | `optimizer.step([emb_module])` |
| 5 | 梯度应用 | KVClient | `client.update(table_name, ids, grads)` |
| 6 | C++ 接口 | `src/framework/op.h` | `op->EmbUpdate(keys, grads)` |
| 7 | PS 客户端 | `src/ps/client/` | Get → Update → Put |
| 8 | 参数更新完成 | - | PS 中的嵌入向量更新 |

### 异步预取流程

| 步骤 | 组件 | 文件路径 | 代码/操作 |
|------|------|---------|----------|
| 1 | 模型 | 用户代码 | `prefetch_id = client.prefetch(ids)` |
| 2 | KVClient | `src/python/pytorch/recstore/KVClient.py` | 返回唯一 prefetch_id |
| 3 | 后台读取 | `src/framework/pytorch/op_torch.cc` | `op->EmbPrefetch()` 异步执行 |
| 4 | 计算重叠 | 用户代码 | 执行其他计算，不阻塞 |
| 5 | 等待完成 | KVClient | `result = client.wait_for_prefetch(prefetch_id)` |
| 6 | 获取结果 | C++ 侧 | `op->GetPretchResult()` |
| 7 | 返回 | PyTorch | `[N, embedding_dim]` 张量 |

## 模块说明

详细文档：

- [op_interface.md](./op_interface.md) - CommonOp 接口定义
- [kvlient.md](./kvclient.md) - KVClient Python 客户端
- [pytorch_binding.md](./pytorch_binding.md) - PyTorch C++ 扩展
- [embedding_modules.md](./embedding_modules.md) - EmbeddingBag / DistEmbedding
- [optimizer.md](./optimizer.md) - 优化器与梯度更新

## 配置示例

### 初始化嵌入表

```python
kv_client = RecStoreClient(library_path="/path/to/lib_recstore_ops.so")

# 初始化嵌入表
kv_client.init_data(
    name="user_embedding",
    shape=(1000000, 128),  # num_embeddings, embedding_dim
    dtype=torch.float32,
    init_func=lambda shape, dtype: torch.randn(shape, dtype=dtype) * 0.01
)
```

### 创建 EmbeddingBag

```python
emb_configs = [
    # ...
]

emb_bag = RecStoreEmbeddingBagCollection(
    embedding_bag_configs=emb_configs,
    lr=0.01,
    enable_fusion=True,
    fusion_k=30
)
```
