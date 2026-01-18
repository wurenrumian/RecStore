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

```bash
# 安装 PyTorch with cxx11abi
cd binary
# pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-2.5.0a0+git*.whl

# 初始化环境依赖
cd ../dockerfiles
bash init_env_inside_docker.sh > init_env.log 2>&1
```

???+ note "PyTorch with cxx11abi"
    RecStore 项目依赖于 PyTorch with cxx11abi，需要自行构建，你可以参考 [项目提供的脚本](dockerfiles/build_torch_wheel.sh) 来安装。

## 6. 编译 RecStore

最后，编译项目：

```bash
cd ..
mkdir build
cd build

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.10 ..

make -j
```
