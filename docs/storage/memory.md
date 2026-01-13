# 内存管理层

## 概述

内存管理层位于 `src/memory/`，为 KV 引擎提供持久化内存分配和管理功能。核心接口是 `MallocApi`。

## 核心接口

### MallocApi

所有内存管理器的抽象基类：

```cpp
class MallocApi {
  virtual char* New(int memory_size) = 0;
  virtual bool Free(void* memory_data) = 0;
};
```

| 方法 | 参数 | 说明 |
|------|------|------|
| New | memory_size | 分配指定字节数的内存 |
| Free | memory_data | 释放已分配的内存 |

## 内存管理器实现

### PersistSimpleMalloc

最简单的持久化内存分配器，基于 slab 机制。

**特点**
- 预分配固定大小的 slab
- 使用位图管理空闲块
- 线程安全 (通过 base::Lock)

**成员变量**

| 成员 | 类型 | 说明 |
|------|------|------|
| shm_file_ | ShmFile | 共享内存文件映射 |
| nr_chunks_ | int64 | 总 chunk 数量 |
| chunks_ | vector<Chunk*> | 所有 chunk 列表 |
| free_chunks_ | deque<Chunk*> | 空闲 chunk 队列 |
| size_to_chunks_ | unordered_map | size 到 chunk 队列的映射 |

**Chunk 结构**

每个 chunk 对应一个固定大小的 slab：

| 字段 | 说明 |
|------|------|
| offset | 在共享内存中的偏移 |
| size | 分配单元大小 |
| capacity | 可分配的单元数量 |
| bits | 位图，标记已使用单元 |

### PersistLoopShmMalloc

基于循环首次适应算法的持久化内存分配器。

**特点**
- 循环查找空闲块，具有 LRU 特性
- 支持快速分配路径 (FastMalloc)
- 使用位图 + 块大小数组管理
- 支持恢复机制

**成员变量**

| 成员 | 类型 | 说明 |
|------|------|------|
| shm_file_ | ShmFile | 共享内存文件 |
| data_ | char* | 数据区起始指针 |
| used_bits_ | uint64* | 使用标记位图 |
| block_num_ | int64 | 总块数 (每块 8 字节) |
| last_malloc_block_ | int64 | 上次分配位置 |
| total_used_ | int64 | 已使用块数 |
| healthy_used_ | int64 | 健康阈值 |

**分配策略**

1. **FastMalloc** - 针对常见大小 (< max_fast_list_type)
   - 维护空闲链表 `fast_malloc_[size]`
   - O(1) 分配和释放
   - 可通过 `DisableFastMalloc()` 关闭

2. **LoopMalloc** - 循环首次适应
   - 从 `last_malloc_block_` 开始查找
   - 找到足够大的空闲块即分配
   - 具有 LRU 特性，自动淘汰冷数据

**块管理**

| 操作 | 说明 |
|------|------|
| BlockNum(size) | 计算需要多少个 8 字节块 |
| BlockIndex(offset) | 将偏移转换为块索引 |
| Used(index) | 检查块是否已使用 |
| UseBlock(index) | 标记块为已使用 |
| FreeBlock(index) | 标记块为空闲 |

**方法**

| 方法 | 说明 |
|------|------|
| New(size) | 分配 size 字节内存 |
| Free(data) | 释放内存 |
| total_used() | 返回已使用块数 |
| total_malloc() | 返回分配次数 |
| Healthy() | 检查是否健康 (未超阈值) |
| GetMallocOffset(data) | 将指针转换为偏移 |
| GetMallocSize(data) | 获取分配的大小 |

**配置**

```cpp
PersistLoopShmMalloc(
  const std::string& filename,   // 共享内存文件路径
  int64 memory_size,              // 总内存大小
  std::string medium = "DRAM"     // 介质类型
);
```

## 工厂注册

使用 `#include "memory/memory_factory.h"` 注册所有内存管理器：

```cpp
FACTORY_REGISTER(MallocApi, PersistSimpleMalloc, ...);
FACTORY_REGISTER(MallocApi, PersistLoopShmMalloc, ...);
```

## 配置示例

在 BaseKVConfig 中指定内存管理器：

```json
{
  "path": "/data/recstore",
  "value_memory_management": "PersistLoopShmMalloc",
  "capacity": 1000000,
  "value_size": 128
}
```

默认使用 `PersistLoopShmMalloc`。

## 使用流程

1. KV 引擎从配置读取 `value_memory_management`
2. 通过工厂创建 MallocApi 实例：
   ```cpp
   std::unique_ptr<MallocApi> malloc_api(
     base::Factory<MallocApi, const std::string&, int64>::NewInstance(
       memory_type, shm_path, memory_size
     )
   );
   ```
3. 调用 `New()` 分配内存，存储值数据
4. 调用 `Free()` 释放不再使用的内存

## 线程安全

两种内存管理器均使用 `base::Lock` 保护内部数据结构：
- PersistSimpleMalloc: 全局锁
- PersistLoopShmMalloc: 全局锁 + FastMalloc 按大小分锁

KV 引擎通过 `tid` 参数实现线程局部内存管理，进一步降低锁竞争。

## 持久化

内存管理器通过 `ShmFile` 实现持久化：
- 将内存映射到文件
- 使用 `clflushopt_range()` 刷新缓存
- 支持恢复机制 (`AddMallocs4Recovery`)
