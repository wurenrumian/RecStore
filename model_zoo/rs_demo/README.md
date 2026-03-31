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

## 4. 结果文件

- JSONL：`/tmp/rs_demo_events.jsonl`
- CSV：`/tmp/rs_demo_embupdate.csv`
- Server 日志：`/tmp/rs_demo_ps_server.log`
