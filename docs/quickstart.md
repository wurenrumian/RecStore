# 快速开始

本指南将帮助你搭建 RecStore 的开发和运行环境，相关步骤出现错误可以查看 [FAQ](./faq.md)。

## 1. 环境准备

RecStore 推荐使用 Docker 进行环境配置。在开始之前，请确保你的系统已安装以下工具：

*   **Docker**: [安装指南](https://docs.docker.com/engine/install/ubuntu/)
*   **NVIDIA Docker (NVIDIA Container Toolkit)**: [安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Ubuntu 快速安装脚本

如果你使用的是 Ubuntu 系统，可以使用以下命令快速安装 Docker 和 NVIDIA Container Toolkit：

```bash
# 1. 安装 Docker
curl -fsSL https://get.docker.com | sudo sh

# 2. 安装 NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

## 2. 获取代码

克隆 RecStore 仓库并更新子模块：

```bash
git clone https://github.com/RecStore/RecStore.git
cd RecStore
git submodule update --init --recursive
```

## 3. 构建 Docker 镜像

进入 `dockerfiles` 目录并构建镜像：

```bash
cd dockerfiles
sudo docker build -f Dockerfile.recstore --build-arg uid=$UID  -t recstore .
cd -
```

## 4. 启动容器

你可以使用以下命令启动容器。**请务必根据你的实际环境修改路径映射 ( `-v` 选项)**。

```bash
RECSTORE_PATH="$(cd .. && pwd)" \
sudo docker run --cap-add=SYS_ADMIN --privileged \
    --security-opt seccomp=unconfined --runtime=nvidia \
    --name recstore --net=host \
    -v ${RECSTORE_PATH}:${RECSTORE_PATH} \
    -v /dev/shm:/dev/shm \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev:/dev \
    -w ${RECSTORE_PATH} \
    --rm -it --gpus all -d recstore
```

或者使用我们提供的脚本（需修改脚本内的变量）：

```bash
cd dockerfiles
bash start_docker.sh
cd -
```

进入容器：

```bash
sudo docker exec -it recstore /bin/bash
```

## 5. 容器内环境初始化

进入容器后，运行以下脚本进行一键初始化：

```bash title="初始化环境依赖"
cd dockerfiles
bash init_env_inside_docker.sh > init_env.log 2>&1

# 安装 GPU 开发默认依赖（全局安装）
bash init_dlrm.sh --mirror=0
```

??? note "默认 GPU 开发版本"
    当前推荐的 GPU 开发环境版本为：

    - `torch==2.7.1+cu118`
    - `torchrec==1.2.0`
    - `fbgemm-gpu==1.2.0`
    - `torchmetrics==1.0.3`

    其中 `torch` 需要满足 `torch.compiled_with_cxx11_abi() == True`。`dockerfiles/init_dlrm.sh` 默认会复用已经安装好的全局 `torch`，并安装与其匹配的 TorchRec 依赖。

    如需手动安装，可直接执行：

    ```bash
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1
    python3 -m pip install torchrec==1.2.0 fbgemm-gpu==1.2.0 torchmetrics==1.0.3 tqdm
    ```

??? note "CI 环境说明"
    CI 当前仍以 CPU-only 为主，用于构建与基础验证。CI 中会继续安装 CPU 版 `torch` 和 `libtorch`，但在无 CUDA toolkit 或 CPU-only `torch` 环境下，`dockerfiles/init_dlrm.sh` 会自动跳过 TorchRec/FBGEMM GPU 依赖安装。

## 6. 编译 RecStore

最后，编译项目：

```bash
cd ..
mkdir build
cd build

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.10 ..

make -j
```
