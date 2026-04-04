# rs_demo

用于在本地快速模拟较大数据量的 RecStore 训练读写更新压力，并导出结构化性能数据。
该 demo 默认复用 DLRM 同源数据入口和组织方式（`processed_day_0_data` + custom dataloader + KJT）。

## 1. 功能

- 使用 DLRM 相同数据来源：`model_zoo/torchrec_dlrm/processed_day_0_data`
- 使用 DLRM 相同稀疏组织：26 特征 -> KJT -> 拼接 ids
- 更新使用与 DLRM 融合模式一致的 fused id：`(table_idx << fuse_k) + id`
- 内部采用模块化结构：`config / data / runtime / runners / cli`
- 执行批量 `emb_read` + `emb_update_table` 循环（可调 steps/batch）
- 可选自动启动/停止 `ps_server`
- 强制开启本地结构化上报（JSONL）
- 自动调用 `analyze_embupdate_stages.py` 导出 CSV
- `read_before_update` 默认走 `emb_prefetch + emb_wait_result` 稳定读路径（避免同步读路径在部分环境下崩溃）

## 2. 快速运行

在仓库根目录执行：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --steps 60 \
  --batch-size 4096 \
  --num-embeddings 200000 \
  --embedding-dim 128 \
  --jsonl /tmp/rs_demo_events.jsonl \
  --csv /tmp/rs_demo_embupdate.csv
```

TorchRec backend（不启动 ps_server）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server \
  --torchrec-main-csv /tmp/rs_demo_torchrec_main.csv \
  --torchrec-main-agg-csv /tmp/rs_demo_torchrec_main_agg.csv
```

如需 profiler trace（每次运行会在 trace dir 下生成多个 trace 文件，并聚合到 trace csv）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server \
  --torchrec-profiler \
  --torchrec-trace-dir /tmp/rs_demo_torchrec_traces \
  --torchrec-main-csv /tmp/rs_demo_torchrec_main.csv \
  --torchrec-main-agg-csv /tmp/rs_demo_torchrec_main_agg.csv \
  --torchrec-trace-csv /tmp/rs_demo_torchrec_trace.csv

# 可选：同配置下已有 RecStore CSV 时，导出对照差值表
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server \
  --torchrec-main-csv /tmp/rs_demo_torchrec_main.csv \
  --torchrec-main-agg-csv /tmp/rs_demo_torchrec_main_agg.csv \
  --torchrec-compare-recstore-csv /tmp/rs_demo_embupdate.csv \
  --torchrec-compare-csv /tmp/rs_demo_recstore_torchrec_compare.csv
```

## 3. 常用参数

- `--num-embeddings`：表大小
- `--embedding-dim`：向量维度
- `--batch-size`：每步 keys 数
- `--steps`：总迭代数
- `--warmup-steps`：预热步数（不计入脚本内 read/update 统计）
- `--data-dir`：DLRM processed day0 目录（默认 `model_zoo/torchrec_dlrm/processed_day_0_data`）
- `--fuse-k`：与 DLRM 相同的融合位移参数（默认 `30`）
- `--read-before-update/--no-read-before-update`：是否每步先读后更
  - 开启时：读路径采用 `prefetch/wait`，并统计 `emb_read` 耗时
- `--start-server/--no-start-server`：是否自动起停 `ps_server`
- `--server-port0/--server-port1`：server 端口（默认读取 `recstore_config.json`）
- `--allocator`：value 内存管理器（默认 `R2ShmMalloc`，更适合压测）
- `--torchrec-main-csv`：TorchRec 主报表 CSV 路径
- `--torchrec-main-agg-csv`：TorchRec 主报表聚合 CSV 路径（mean/p50/p95/max）
- `--torchrec-profiler`：启用 Torch profiler 并导出 trace 聚合 CSV
- `--torchrec-trace-dir`：Torch profiler trace 输出目录
- `--torchrec-trace-csv`：Torch profiler trace 聚合 CSV 路径
- `--torchrec-compare-recstore-csv`：可选，指定 RecStore CSV 以导出对照差值表
- `--torchrec-compare-csv`：RecStore vs TorchRec 对照差值 CSV 路径

## 4. 结果文件

- JSONL：`/tmp/rs_demo_events.jsonl`
- CSV：`/tmp/rs_demo_embupdate.csv`
- Server 日志：`/tmp/rs_demo_ps_server.log`
- TorchRec 主报表 CSV：`/tmp/rs_demo_torchrec_main.csv`
- TorchRec 主报表聚合 CSV：`/tmp/rs_demo_torchrec_main_agg.csv`
- TorchRec profiler trace CSV：`/tmp/rs_demo_torchrec_trace.csv`
- RecStore vs TorchRec 对照 CSV：`/tmp/rs_demo_recstore_torchrec_compare.csv`

TorchRec 主报表（`--torchrec-main-csv`）关键列：

- `collective_total_ms`：collective launch + wait 的总耗时
- `kv_local_only_ms`：本地 embedding lookup + pool 的耗时（不含 pack/unpack）
- `kv_extended_ms`：输入打包 + 本地 lookup/pool + 输出解包的总耗时
- `network_proxy_torchrec_extended_ms`：`collective_total + input_pack + output_unpack` 的扩展通信代理项

TorchRec 主报表聚合 CSV（`--torchrec-main-agg-csv`）会对每个 `*_ms` 列导出：

- `*_mean`
- `*_p50`
- `*_p95`
- `*_max`

对照差值 CSV（`--torchrec-compare-csv`）默认导出以下口径：

- `network_main`：`RecStore(network_transport)` vs `TorchRec(collective_total)`
- `network_extended`：`RecStore(network_transport)` vs `TorchRec(collective + pack + unpack)`
- `kv_strict`：`RecStore(storage_backend_update)` vs `TorchRec(kv_local_only)`
- `server_vs_extended`：`RecStore(server_total)` vs `TorchRec(kv_extended)`

`collective_mode=not_measured_single_process` 表示当前仅单进程运行，未采集到多进程 collective 统计。
