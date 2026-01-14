# RecStore EBC 精度对齐测试

**代码位置**: 

*   **单机**: `src/python/pytorch/recstore/unittest/test_ebc_precision.py`
*   **多机/多进程**: `src/python/pytorch/recstore/unittest/test_ebc_precision_multiprocess.py`

## 简介
精度对齐测试验证分布式系统可靠性，将 **RecStore EBC** (Client + Server) 的运行结果与 **PyTorch/TorchRec 原生 EBC** (纯本地内存) 进行逐位比对，确保分布式化没有引入任何计算误差。

## 测试场景概览

我们提供了两种维度的测试：

| 场景 | 对应文件 | 描述 | 核心挑战 |
| :--- | :--- | :--- | :--- |
| **单进程 (Single Process)** | `test_ebc_precision.py` | 在一个进程内同时运行 RecStore 和 Standard EBC。 | 验证基本的 Forward/Backward 数学正确性。 |
| **多进程 (Multi Process)** | `test_ebc_precision_multiprocess.py` | 模拟多张卡 (Rank) 同时训练，每张卡维护部分 Key。 | 验证 **Rank 间的数据隔离** 以及 **梯度的并发更新**。 |

## 测试流程

### 单进程

| 步骤 | 逻辑描述 | 代码片段示意 |
| :--- | :--- | :--- |
| **权重同步** | 启动前及每个 Epoch 后，强制同步两者权重。 | `kv_client.emb_write(keys, standard_ebc.weights)` |
| **Forward** | 相同 Input -> 比较 Output。 | `assert allclose(std_out, rec_out)` |
| **Backward** | `loss.backward()` -> 验证梯度 Trace 是否生成。 | `assert len(recstore_ebc._trace) > 0` |
| **Update** | `SparseSGD.step()` -> 比较更新后的权重。 | `assert allclose(new_std_weight, new_rec_weight)` |

### 多进程协同

模拟真实分布式训练环境（DDP模式），验证多个 Worker 同时读写参数服务器时的行为。

*   **启动方式**: 使用 `subprocess` 或 `mp.spawn` 启动多个 Worker 进程。
*   **数据切分**: 每个 Rank 只负责一部分 Embedding Key（例如 Rank 0 负责 0-1000, Rank 1 负责 1000-2000）。

| 步骤 | 分布式特有逻辑 | 预期行为 |
| :--- | :--- | :--- |
| **独立初始化** | 只有 Rank 0 负责初始化 Table，其他 Rank 等待 Barrier。 | 避免重复建表导致的 Race Condition。 |
| **局部视角验证** | 每个 Rank 只 `pull` 自己负责的那一段 Key 进行比对。 | `ids = range(rank*N, (rank+1)*N)`<br>`verify(pull(ids), local_slice)` |
| **并发 Update** | 所有 Rank 同时发送梯度更新。 | Server 端应能正确处理并发请求，不发生锁冲突或数据错乱。 |
