# KVEngine 配置与命名清理说明（草案）

## 当前状态（2026-03-29）

已完成第一阶段落地（兼容旧字段）：

1. `ResolveEngine` 新增模式字段 `mode=STATIC|DYNAMIC` 与组合校验（显式配置时严格校验）。
2. 配置命名兼容：
   - `index_medium` <-> `index_type`
   - `value_medium` <-> `value_type`
   - `DRAM_SIZE` <-> `shmcapacity`
   - `SSD_SIZE` <-> `ssdcapacity`
3. 分配器命名兼容：
   - `allocator_type=PERSIST_LOOP_SLAB|R2_SLAB`
   - 兼容旧字段 `value_memory_management=PersistLoopShmMalloc|R2ShmMalloc`
   - 公共实现入口迁移到 `src/memory/allocators/`（统一工厂与 Slab 封装）
4. `value_layout=FIXED|VARIABLE`：
   - 当前仅允许 `VARIABLE + HYBRID(ValueManager)`，其余组合启动期报错。
5. 为 SSD index 默认补齐 `io_backend_type/queue_cnt/page_id_offset/file_path`，减少漏配风险。

## 背景

当前 KVEngine 的配置语义和命名存在几个问题：

1. 静态/动态两类能力混在一起，配置边界不清晰。  
2. `shmcapacity` / `ssdcapacity` 等字段可读性一般，不利于新同学理解。  
3. slab 分配器实现分散，配置名称不直观。  
4. “变长 value 支持”只在部分路径可用，但文档和配置未显式表达。

本文档用于把需求统一成可实施的重构目标，先对齐语义，再推进代码改造。

## 目标

1. 明确 KVEngine 的“静态模式”和“动态模式（带缓存策略，如 LRU）”。
2. 统一配置命名，让配置读起来就能知道资源归属（DRAM/SSD）。
3. 收敛 slab 分配器实现到公共目录，并支持通过配置显式切换。
4. 明确“变长 value”的支持范围与限制，避免误用。

## 模式与组合

### 1) 静态模式（不含缓存策略）

静态模式定义：无热冷迁移策略、无 LRU/LFU 等动态缓存行为，写入位置和生命周期固定。

支持组合：

| 组合 | 索引介质 | 值介质 | 说明 |
|---|---|---|---|
| S1 | DRAM | DRAM | 纯内存路径 |
| S2 | DRAM | SSD | DRAM 索引 + SSD 值 |
| S3 | SSD | SSD | 全 SSD 路径 |

### 2) 动态模式（接入缓存策略）

动态模式定义：接入 LRU 等缓存策略，允许热冷分层、迁移和淘汰。

支持组合：

| 组合 | 索引介质 | 值介质 | 说明 |
|---|---|---|---|
| D1 | DRAM | HYBRID | DRAM 索引 + 混合值层 |
| D2 | HYBRID | HYBRID | 混合索引 + 混合值层（目标态） |

> 说明：`HYBRID` 指 DRAM + SSD 分层能力，而不是单一介质。

## 变长 value 支持边界

当前约束：

- 仅使用 `ValueManager` 的路径支持变长 value。  
- 非 `ValueManager` 路径视为定长 value 路径。

建议在配置中显式体现：

- `value_layout = "FIXED" | "VARIABLE"`  
- 当 `value_layout = "VARIABLE"` 时，启动时强校验 `ValueManager` 已启用，否则直接报错。

## slab 分配器清理目标

### 1) 代码位置整理

将两种 slab 分配器实现迁移到公共目录，建议：

- `src/memory/allocators/`
  - `slab_allocator_*.h/.cc`
  - `allocator_factory.h/.cc`

并通过统一工厂对外暴露，避免 KVEngine/ValueManager 各自持有分支逻辑。

### 2) 配置可读性优化

现有字段：`shmcapacity`, `ssdcapacity`  
建议统一为：

- `DRAM_SIZE`：DRAM 层容量（字节）  
- `SSD_SIZE`：SSD 层容量（字节）

并补充分配器配置：

- `allocator_type = "PERSIST_LOOP_SLAB" | "R2_SLAB"`

## 建议配置结构（示例）

```json
{
  "base_kv_config": {
    "mode": "STATIC",
    "index_medium": "DRAM",
    "value_medium": "SSD",
    "value_layout": "FIXED",
    "allocator_type": "PERSIST_LOOP_SLAB",
    "DRAM_SIZE": 10737418240,
    "SSD_SIZE": 107374182400
  }
}
```

动态模式示例：

```json
{
  "base_kv_config": {
    "mode": "DYNAMIC",
    "index_medium": "DRAM",
    "value_medium": "HYBRID",
    "cache_policy": "LRU",
    "value_layout": "VARIABLE",
    "allocator_type": "R2_SLAB",
    "DRAM_SIZE": 10737418240,
    "SSD_SIZE": 107374182400
  }
}
```

## 兼容策略（避免一次性破坏）

建议分两阶段：

1. **兼容阶段**  
   - 继续接受旧字段（`shmcapacity`, `ssdcapacity`, `index_type`, `value_type`）。  
   - 内部转换为新字段并打印一次迁移告警。  

2. **收敛阶段**  
   - 文档、示例、测试全部切到新字段。  
   - 旧字段进入弃用列表，后续版本删除。

## 代码改造清单（建议）

1. 抽象配置解析层：新增“旧字段 -> 新字段”转换与校验。  
2. 引擎选择逻辑改为基于 `mode/index_medium/value_medium` 的显式映射。  
3. 分配器工厂化：从业务代码中移除分配器分支。  
4. 变长约束前置：启动时校验 `value_layout` 与 `ValueManager` 组合。  
5. 补齐测试矩阵：S1/S2/S3 + D1/D2 + FIXED/VARIABLE + 两种 allocator。  

## 非目标（本轮不做）

1. 不在本轮引入新缓存策略算法（只做接入框架和配置抽象）。  
2. 不改变既有读写语义与一致性策略。  
3. 不做大规模存储格式迁移。

## 这次改动“本质上在做什么”

一句话：  
**把“引擎组合能力”和“资源/分配策略配置”从历史命名中解耦，变成可读、可验证、可演进的配置契约。**
