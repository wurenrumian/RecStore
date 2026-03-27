# 快速运行

本指南用于在完成[搭建指南](quickstart.md)后，进行全流程的运行，相关步骤出现错误可以查看 [FAQ](./faq.md)。

## 1. 启动参数服务器

在仓库根目录执行：

```bash title="自动读取配置文件，启动ps客户端"
./buidl/bin/ps_server
```

## 2. 运行计算层模型

DLRM 使用的数据集为 [Criteo Kaggle Display Advertising Challenge Dataset](https://ailab.criteo.com/ressources/)，项目切片了第 0 天的数据方便进行测试和分析，你可以在 [day_0.csv](https://github.com/user-attachments/files/23355355/day_0.csv) 下载前 4096 条的数据。需要把数据去掉后缀放到 `model_zoo/torchrec_dlrm/partial_data` 下，然后运行：

```bash title="预处理./partial_data中的数据"
bash scripts/process_single_day.sh ./partial_data ./processed_day_0_data > process.log 2>&1
```

来完成数据集的加载，随后可以直接进行训练。

在运行前，请确认全局 Python 环境已经安装：

```bash
python3 -c "import torch, torchrec, fbgemm_gpu, torchmetrics; print(torch.__version__, torch.version.cuda, torch.compiled_with_cxx11_abi())"
```

推荐输出：

```text
2.7.1+cu118 11.8 True
```

=== "RecStore"
    ```bash
    cd model_zoo/torchrec_dlrm/
    bash run_single_day.sh
    ```
=== "TorchRec"
    ```bash
    cd model_zoo/torchrec_dlrm/
    bash run_single_day.sh --torchrec
    ```

可以使用 `--help` 参数来获取支持的所有参数。
