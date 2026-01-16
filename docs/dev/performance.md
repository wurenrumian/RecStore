# 性能分析

RecStore 内置了完善的性能埋点与分析机制，涵盖了从 PyTorch OP 层、C++ 客户端到 gRPC 服务端及底层存储的完整链路。

## 1. 快速使用

### 运行性能统计

**服务端 (PS Server)**

可以通过环境变量 `CPUPROFILE` 开启 CPU Profiler，并通过 `--perf_report_path` 指定性能报告输出路径：

```bash
CPUPROFILE=/tmp/ps_cpu.prof ./build/bin/ps_server --perf_report_path=/tmp/ps_perf.log
```

**训练端 (DLRM Trainer)**

使用 `run_single_day.sh` 脚本一键运行性能测试：

```bash
export RECSTORE_PERF_REPORT_PATH=/tmp/trainer_perf.log
export RECSTORE_PERF_INTERVAL_MS=5000
bash run_single_day.sh --custom --dataset-size 4096 --epochs 10
```

???+ note "关于run_single_day 脚本"
    这是对于模型层 DLRM 的小型测试脚本，可以使用单天数据进行测试，同时限制了数据量和嵌入表的大小。
    运行 `bash run_single_day.sh --help` 可查看更多参数说明。

### 查看结果

**日志分析**

`perf_report_path` 指定的文件会实时记录每个时间窗口（如 5000ms）内的耗时统计（P50/P99）：

```bash
# 查看服务端性能概览
tail -n 50 /tmp/ps_perf.log

# 查看训练端性能概览
tail -n 50 /tmp/trainer_perf.log
```

**CPU Profiling 分析**

使用 `google-pprof` 分析 CPU 热点（需要安装 `gperftools`）：

```bash
google-pprof --text build/bin/grpc_ps_server /tmp/ps_cpu.prof
# 或者导出 pdf/svg
google-pprof --pdf build/bin/grpc_ps_server /tmp/ps_cpu.prof > ps_cpu.pdf
```

**模型层统计**

```bash
tensorboard --logdir=./logs --port=6006
```

## 2. 性能埋点关键字

RecStore 使用 `xmh::Timer` (位于 `src/base/timer.h`) 进行全链路的耗时打点。数据流向如下：

### 写路径自顶向下

| 层级 (Layer) | Timer 名称 | 说明 | 代码位置 |
| :--- | :--- | :--- | :--- |
| **PyTorch OP** | `OP.EmbWrite.Total` | Python 端调用 C++ OP 的总耗时 | `src/framework/pytorch/op_torch.cc` |
| | `OP.EmbWrite.Call` | 核心逻辑调用耗时 | |
| **C++ Client** | `ClientOp.EmbWrite.Total` | 客户端操作总耗时 | `src/framework/op.cc` |
| | `ClientOp.EmbWrite.BuildVector` | Tensor 转 C++ Vector 的开销 | |
| **gRPC Client** | `Client.PutParameter.Total` | RPC 请求总耗时 | `src/ps/grpc/dist_grpc_ps_client.cpp` |
| | `Client.PutParameter.Serialize` | 序列化 Data Request 耗时 | |
| | `Client.PutParameter.RPC` | 网络传输 + 服务端处理总耗时 | |
| **Server** | `PS.PutParameter.Handle` | 服务端处理总耗时 | `src/ps/grpc/grpc_ps_server.cpp` |
| | `PS.PutParameter.KVPutAll` | 写入底层 KV 存储的耗时 | |

### 读路径自顶向下

| 层级 (Layer) | Timer 名称 | 说明 | 代码位置 |
| :--- | :--- | :--- | :--- |
| **PyTorch OP** | `OP.EmbRead.Total` | Python 端调用 C++ OP 的总耗时 | `src/framework/pytorch/op_torch.cc` |
| | `OP.EmbRead.ToCPUKeys` | Embedding Key 从 GPU 拷贝到 CPU 的耗时 | |
| **gRPC Client** | `Client.GetParameter.Total` | RPC 请求总耗时 | `src/ps/grpc/dist_grpc_ps_client.cpp` |
| | `Client.GetParameter.AsyncWait` | 异步等待网络回包的耗时（反映服务端处理+网络延迟） | |
| **Server** | `KV BatchGet` | 底层 KV 引擎批量读取耗时 | `src/storage/kv_engines.md` |

??? tip "什么是 Timer?"
    `xmh::Timer` 是一个高性能的 C++ 计时器工具类。它通过 `Timer::Start("Name")` 和 `Timer::Stop("Name")` 记录代码块耗时，并自动统计 P50、P99 等分位数值，定期输出到日志文件中。
