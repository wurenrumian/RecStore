# DistEmbedding 模块测试

**代码位置**: `src/python/pytorch/recstore/unittest/test_dist_emb.py`

## 简介
`DistEmbedding` 是 RecStore 的核心构建块，它是 `nn.Module` 的直接子类。此测试验证它完全遵循 PyTorch 的 Autograd 机制，并能无缝接入 Parameter Server。

## 流程与测试点

| 测试方法 | 描述 | 图解/逻辑 | 预期结果 |
| :--- | :--- | :--- | :--- |
| `test_initialization` | **初始化与元数据注册** | `DistEmbedding(...)` -> 连接 Client -> 注册 Table | Client 内部元数据结构中应包含该 Table 的 Shape 和 Dtype。 |
| `test_backward` | **梯度追踪 (Tracing)** | `loss.backward()` -> **Hook 触发** -> 记录 `{Keys, Grads}` 到 Trace | Backward 仅记录 Trace，**不更新**权重。权重在此时应保持不变。 |
| `test_optimizer_step` | **优化器更新** | `SparseSGD.step()` -> 读取 Trace -> `client.push` | 调用 `step()` 后，权重发生变化，且符合 SGD 更新公式。 |
| `test_duplicate_ids` | **梯度聚合** | 输入 Batch: `[ID 1, ID 1, ID 1]`<br>Backward: `Wait...` | Server 端收到的 ID 1 的梯度应为 3 次梯度的**总和**。 |
| `test_persistence` | **数据持久化** | `Emb(name="A")` 退出 -> `Emb(name="A")` 重启 | 新实例应能读取到上一个实例写入的数据。 |
