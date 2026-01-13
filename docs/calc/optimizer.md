# 优化器与梯度更新

## 概述

RecStore 采用后端优化架构：**所有优化器逻辑（SGD、AdaGrad、RowWiseAdaGrad）在参数服务器后端实现**。Python 侧仅负责梯度收集与传输，通过 `KVClientOp::EmbUpdate` 将梯度发送到 PS，由后端在存储层完成参数更新。

## 架构

| 阶段 | 模块 | 文件路径 | 说明 |
|------|------|---------|------|
| 1 | 前向传播 | `src/python/pytorch/torchrec/EmbeddingBag.py` | 接收特征，调用 forward() |
| 2 | EmbeddingBag | `RecStoreEmbeddingBagCollection.forward()` | 执行 _RecStoreEBCFunction.forward |
| 3 | 梯度计算 | `loss.backward()` | PyTorch 自动求导，触发 backward |
| 4 | 梯度追踪 | `_DistEmbFunction.backward()` | 收集 (ID, grad) 对到 `_trace` |
| 5 | 梯度发送 | `SparseOptimizer.step()` | 遍历 _trace，调用 KVClient.update |
| 6 | 后端更新 | `KVClientOp.EmbUpdate()` | 将梯度发送到 PS |
| 7 | 优化器应用 | `src/optimizer/optimizer.cpp` | PS 后端执行 SGD/AdaGrad 等更新 |

## 梯度追踪机制

### RecStoreEmbeddingBagCollection 梯度追踪

反向传播时，_RecStoreEBCFunction.backward 将梯度追踪到模块的 _trace：

**追踪流程**

| 步骤 | 操作 | 输出 |
|------|------|------|
| 1 | `for i, key in enumerate(feature_keys)` | 遍历每个特征 |
| 2 | `config_name = module._config_names[key]` | 获取嵌入表名 |
| 3 | `ids_to_update = values_cpu[start:end]` | 提取该特征的 ID |
| 4 | `grad_for_bag = grad_output_reshaped[sample_idx, i]` | 提取该特征的梯度 |
| 5 | `module._trace.append((config_name, ids, grads))` | 追踪到 _trace |

**追踪内容**

| 字段 | 类型 | 说明 |
|------|------|------|
| config_name | str | 嵌入表名称 |
| ids_to_update | Tensor | int64, [N] 的 ID |
| grads_to_trace | Tensor | float32, [N, D] 的梯度 |

### DistEmbedding 梯度追踪

_DistEmbFunction.backward 对重复 ID 的梯度进行合并：

**梯度合并流程**

| 步骤 | 代码位置 | 代码 | 说明 |
|------|---------|------|------|
| 1 | `src/python/pytorch/recstore/DistEmb.py:_DistEmbFunction.backward` | `unique_ids, inverse = torch.unique(ids_cpu, return_inverse=True)` | 对 ID 去重，保留逆映射 |
| 2 | 同上 | `grad_sum = torch.zeros((unique_ids.size(0), grad_cpu.size(1)))` | 创建去重后的梯度容器 |
| 3 | 同上 | `grad_sum.index_add_(0, inverse, grad_cpu)` | 相同 ID 的梯度累加 |
| 4 | 同上 | `module_instance._trace.append((unique_ids, grad_sum))` | 追踪合并后的梯度 |

## SparseOptimizer

**Python 侧梯度收集与传输组件**（不执行参数更新）

### 接口

```python
class SparseOptimizer:
    def step(self, modules: List[torch.nn.Module]):
        """处理所有模块的梯度追踪"""
        for module in modules:
            if hasattr(module, '_trace'):
                self._apply_gradients(module)
    
    def _apply_gradients(self, module):
        """应用模块内记录的梯度"""
```

### 工作流程

| 步骤 | 代码位置 | 代码 | 说明 |
|------|---------|------|------|
| 1 | 用户代码 | `optimizer = SparseOptimizer(kv_client, lr=0.01)` | 初始化优化器 |
| 2 | 模型 | `output = emb_module(features)` | 前向传播，执行嵌入查询 |
| 3 | 用户代码 | `loss.backward()` | 反向传播，收集梯度到 _trace |
| 4 | `src/python/pytorch/recstore/optimizer.py` | `optimizer.step([emb_module])` | 遍历 _trace，调用 EmbUpdate 发送梯度 |
| 5 | `src/optimizer/optimizer.cpp` | `SGD::Update() / AdaGrad::Update()` | PS 后端执行实际参数更新 |
| 6 | `src/python/pytorch/recstore/optimizer.py` | `emb_module.reset_trace()` | 清理追踪，为下一个 batch 做准备 |

## 后端优化器实现

### KVClientOp::EmbUpdate

**梯度传输接口**

```cpp
virtual void EmbUpdate(
    const RecTensor& keys, 
    const RecTensor& grads
) = 0;
```

| 参数 | 说明 |
|------|------|
| keys | uint64 张量 [N] |
| grads | float32 张量 [N, D] |

**功能**: 将梯度通过 RPC 发送到参数服务器，不在客户端执行参数更新

### 后端优化器

**实现位置**: `src/optimizer/optimizer.cpp`

**支持的优化器**

| 优化器 | 类名 | 更新公式 | 说明 |
|--------|------|---------|------|
| SGD | `SGD::Update()` | `emb -= lr * grad` | 标准随机梯度下降 |
| AdaGrad | `AdaGrad::Update()` | `sum_sq += grad^2`<br>`emb -= lr * grad / sqrt(sum_sq + eps)` | 自适应学习率，每个参数独立累积 |
| RowWiseAdaGrad | `RowWiseAdaGrad::Update()` | `sum_sq_row += sum(grad^2)`<br>`emb -= lr * grad / sqrt(sum_sq_row + eps)` | 行级自适应，每行共享累积量 |

**更新流程**

| 步骤 | 代码位置 | 说明 |
|------|---------|------|
| 1 | `src/ps/base/base_ps_server.cpp` | PS 接收 UpdateParameter RPC |
| 2 | `src/optimizer/optimizer.cpp:SGD::Update()` | 从 BaseKV 读取当前参数 |
| 3 | 同上 | 应用优化器公式更新参数 |
| 4 | 同上 | 将更新后的参数写回 BaseKV |

## 学习率配置

学习率在初始化嵌入表时通过配置传递给后端优化器

### 配置方式

| 方式 | 代码位置 | 说明 |
|------|---------|------|
| 配置文件 | `recstore_config.json` | 在 `embedding_tables` 中指定 `learning_rate` |
| 初始化 | `KVClient.init_embedding_table()` | 通过 `EmbeddingTableConfig` 传递学习率 |
| 后端存储 | `src/optimizer/optimizer.cpp:Init()` | 优化器初始化时保存每个表的学习率 |

## 梯度传输与聚合

### Python 侧梯度收集

梯度追踪在 Python 内存中进行，不涉及参数更新

```python
_trace = [
    (table_name, ids, grads),
    (table_name, ids, grads),
    ...
]
```

### 批量传输

SparseOptimizer 逐条处理 _trace，每个 (ids, grads) 通过 `KVClientOp.EmbUpdate` 发送到 PS

| 步骤 | 说明 |
|------|------|
| 1 | `SparseOptimizer.step()` 遍历 `_trace` |
| 2 | 调用 `kv_client.update(table_name, ids, grads)` |
| 3 | `KVClientOp::EmbUpdate` 通过 gRPC/BRPC 发送到 PS |
| 4 | PS 后端优化器执行参数更新 |

### 后端批量更新

PS 接收到梯度后，`src/optimizer/optimizer.cpp` 批量处理：

```cpp
void SGD::Update(string table, const vector<uint64_t>& keys,
                 const vector<vector<float>>& grads, unsigned tid) {
    vector<vector<float>> current_values(keys.size());
    base_kv_->Get(keys, current_values);
    
    for (size_t i = 0; i < keys.size(); i++) {
        for (size_t j = 0; j < grads[i].size(); j++) {
            current_values[i][j] -= learning_rate_ * grads[i][j];
        }
    }
    
    base_kv_->Put(keys, current_values);
}
```

## 优化器状态管理

### 状态存储位置

**AdaGrad/RowWiseAdaGrad 的累积梯度平方和存储在后端**

| 优化器 | 状态 | 存储位置 | 数据结构 |
|--------|------|---------|----------|
| SGD | 无 | - | 仅存储参数本身 |
| AdaGrad | `sum_squared_gradients` | `src/optimizer/optimizer.cpp` | `unordered_map<uint64_t, vector<float>>` |
| RowWiseAdaGrad | `sum_squared_gradients_row` | 同上 | `unordered_map<uint64_t, float>` |

### 初始化流程

| 步骤 | 代码位置 | 说明 |
|------|---------|------|
| 1 | `src/ps/base/base_ps_server.cpp` | 接收 InitEmbeddingTable RPC |
| 2 | `src/optimizer/optimizer.cpp:Init()` | 为嵌入表分配优化器状态存储 |
| 3 | 同上 | 初始化学习率、epsilon 等超参数 |
