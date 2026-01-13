# KVClient Python 客户端

## 概述

`RecStoreClient` 是 Python 侧的单例客户端，负责加载 C++ 操作库并提供 Python API。位于 `src/python/pytorch/recstore/KVClient.py`。

## 初始化

### 单例模式

```python
client = RecStoreClient(library_path="/path/to/lib_recstore_ops.so", role="default")
```

| 参数 | 说明 |
|------|------|
| library_path | C++ 扩展库路径，默认从 build 目录查找 |
| role | 客户端角色，默认 "default" |

**特点**

* 单例模式确保全局只有一个客户端实例
* 重复初始化时直接返回已有实例
* 支持多个角色的客户端区分

## 数据管理 API

### 初始化嵌入表

```python
def init_data(
    name: str,
    shape: Tuple[int, int],
    dtype: torch.dtype,
    part_policy: Any = None,
    init_func: Optional[Callable] = None,
    is_gdata: bool = True,
    base_offset: int = 0
):
```

| 参数 | 说明 |
|------|------|
| name | 嵌入表名称 (唯一) |
| shape | (num_embeddings, embedding_dim) |
| dtype | 数据类型，通常 torch.float32 |
| part_policy | 分区策略 (当前未使用) |
| init_func | 初始化函数，签名 (shape, dtype) → Tensor |
| is_gdata | 是否为图数据 (默认 True) |
| base_offset | ID 偏移量 (用于融合) |

**工作流程**

| 步骤 | 代码/操作 | 说明 |
|------|----------|------|
| 1 | 检查 name 是否已初始化 | 避免重复 |
| 2 | `success = self.ops.init_embedding_table(name, num_embeddings, embedding_dim)` | 后端初始化 |
| 3 | `self._tensor_meta[name] = {'shape': shape, 'dtype': dtype}` | 记录元数据 |
| 4 | `initial_data = init_func(shape, dtype) if init_func else torch.zeros(shape)` | 生成初始数据 |
| 5 | `all_keys = torch.arange(shape[0]) + base_offset` | 生成 ID |
| 6 | `self.ops.emb_write(all_keys, initial_data)` | 初始数据写入 |

### 删除嵌入表

```python
def delete_data(name: str):
```

删除已初始化的嵌入表 **(未实现 - 仅清理本地元数据，不清除后端数据)**

### 查询元数据

```python
def get_data_meta(name: str) -> Tuple[torch.dtype, Tuple]:
```

获取已初始化表的数据类型和形状

```python
def data_name_list() -> List[str]:
```

获取所有已初始化的嵌入表名称

```python
def gdata_name_list() -> List[str]:
```

获取所有图数据表名称

## 读写 API

### 拉取 (Pull)

```python
def pull(name: str, ids: torch.Tensor) -> torch.Tensor:
```

批量读取嵌入向量

| 参数 | 说明 |
|------|------|
| name | 嵌入表名称 |
| ids | int64 张量，[N] 形状，嵌入 ID |

| 返回值 | 说明 |
|--------|------|
| values | float32 张量，[N, D] 形状，嵌入向量 |

**工作流程**

| 步骤 | 代码位置 | 代码/操作 | 说明 |
|------|---------|----------|------|
| 1 | `src/python/pytorch/recstore/KVClient.py:pull` | 检查 name 是否已初始化 | 验证表存在 |
| 2 | 同上 | `meta = self._tensor_meta[name]` | 获取 embedding_dim |
| 3 | 同上 | `if ids.dtype != torch.int64: ids = ids.to(torch.int64)` | 确保类型正确 |
| 4 | 同上 | `return self.ops.emb_read(ids, embedding_dim)` | 调用 C++ 扩展 |
| 5 | `src/framework/pytorch/op_torch.cc` | `op->EmbRead(rec_keys, rec_values)` | 后端读取 |
| 6 | `src/python/pytorch/recstore/KVClient.py:pull` | 返回 `[N, embedding_dim]` | 返回向量 |

### 推送 (Push)

```python
def push(name: str, ids: torch.Tensor, data: torch.Tensor):
```

批量写入嵌入向量

| 参数 | 说明 |
|------|------|
| name | 嵌入表名称 |
| ids | int64 张量，[N] 形状 |
| data | float32 张量，[N, D] 形状 |

**工作流程**

| 步骤 | 代码位置 | 代码/操作 | 说明 |
|------|---------|----------|------|
| 1 | `src/python/pytorch/recstore/KVClient.py:push` | 检查 name 有效性 | 验证表存在 |
| 2 | 同上 | `if ids.dtype != torch.int64: ids = ids.to(torch.int64)` | 强制转换 ID 类型 |
| 3 | 同上 | `self.ops.emb_write(ids, data)` | 调用 C++ 扩展 |
| 4 | `src/framework/pytorch/op_torch.cc` | `op->EmbWrite(rec_keys, rec_values)` | 后端写入 |
| 5 | `src/ps/client/` | 与 PS 通信，更新嵌入 | 写入 PS |

## 异步预取 API

### 发起预取

```python
def prefetch(ids: torch.Tensor) -> int:
```

异步预取嵌入向量，返回预取 ID 用于后续查询

| 参数 | 说明 |
|------|------|
| ids | int64 张量，[N] 形状 |

| 返回值 | 说明 |
|--------|------|
| prefetch_id | 用于查询状态的唯一 ID |

### 检查预取状态

```python
def is_prefetch_done(prefetch_id: int) -> bool:
```

检查指定预取是否完成（非阻塞）

### 等待预取完成

```python
def wait_for_prefetch(prefetch_id: int) -> torch.Tensor:
```

阻塞等待预取完成，返回 [N, D] 张量

## 优化器相关 API

### 梯度更新

```python
def update(name: str, ids: torch.Tensor, grads: torch.Tensor):
```

应用梯度更新到嵌入表

| 参数 | 说明 |
|------|------|
| name | 嵌入表名称 |
| ids | int64 张量，[N] 形状 |
| grads | float32 张量，[N, D] 形状 |

**工作流程**

| 步骤 | 代码位置 | 代码/操作 | 说明 |
|------|---------|----------|------|
| 1 | `src/python/pytorch/recstore/KVClient.py:update` | 检查 name 和数据有效性 | 验证输入 |
| 2 | 同上 | `self.ops.emb_update_with_table(name, ids, grads)` | 调用 C++ 扩展 |
| 3 | `src/framework/pytorch/op_torch.cc` | `op->EmbUpdate(table_name, keys, grads)` | 后端更新 |
| 4 | `src/framework/op.h:KVClientOp` | Get 当前嵌入、应用 SGD、Put 回 PS | 参数服务器更新 |

### 学习率设置

```python
def set_learning_rate(lr: float):
```

设置全局学习率 (影响梯度更新)

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| role | str | 客户端角色 |
| client_id | int | 客户端 ID (模拟值) |
| machine_id | int | 机器 ID (模拟值) |
| part_policy | dict | 分区策略 |

## 内部状态

| 成员 | 类型 | 说明 |
|------|------|------|
| _tensor_meta | dict | 嵌入表元数据 (shape, dtype) |
| _full_data_shape | dict | 完整数据形状 |
| _data_name_list | set | 所有初始化表的名称 |
| _gdata_name_list | set | 图数据表的名称 |
| ops | torch.ops.* | 加载的 C++ 操作 |

## 错误处理

| 异常 | 情况 |
|------|------|
| ImportError | 无法加载 C++ 库 |
| RuntimeError | 表未初始化、后端初始化失败 |
| TypeError | 表已存在但形状/dtype 不匹配 |

## 使用示例

```python
import torch
from recstore.KVClient import RecStoreClient

# 初始化客户端
client = RecStoreClient()

# 初始化嵌入表
client.init_data(
    "user_embedding",
    (100000, 64),
    torch.float32,
    init_func=lambda shape, dtype: torch.randn(shape, dtype=dtype) * 0.01
)

# 读取嵌入
ids = torch.tensor([1, 2, 3], dtype=torch.int64)
embeddings = client.pull("user_embedding", ids)  # [3, 64]

# 异步预取
prefetch_id = client.prefetch(ids)
# ... 执行其他计算 ...
result = client.wait_for_prefetch(prefetch_id)  # [3, 64]

# 更新梯度
grads = torch.randn(3, 64, dtype=torch.float32)
client.update("user_embedding", ids, grads)
```
