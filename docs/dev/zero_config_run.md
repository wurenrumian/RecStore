# 免配置环境运行 (Zero-Config Execution)

为了方便开发者快速体验 RecStore 而无需在本地搭建复杂的编译环境，我们提供了基于 GitHub Actions 构建产物的免编译运行方式。

同时，仓库也提供了手动触发的 GitHub Release 工作流，可以把同一套 CPU 产物发布到 **Releases** 页面，方便直接下载固定版本。

???+ Warning "注意"
    该方式仅支持 Linux 环境，以及由于 Actions 环境限制，构建是纯 CPU 版本，可能不支持 GPU 环境。
    
    由于依赖问题，所以对于使用不同版本 GLIBC 的系统，可能需要升级 / 降级版本，

通过下载云端构建好的二进制包，你可以在标准的 Linux 环境（如 Ubuntu 20.04/22.04）中直接运行 RecStore。

## 1. 获取构建产物

1.  进入 RecStore GitHub 仓库的 **[Actions](https://github.com/RecStore/RecStore/actions)** 页面。
2.  点击最新的构建成功（✅ Success）的 Workflow Run（例如 [Build (CPU)](https://github.com/RecStore/RecStore/actions/workflows/ci-build.yml)）。
3.  在页面底部的 **Artifacts** 区域，下载以下文件：
    *   `packed-bundle`: 包含编译好的 `ps_server` 可执行文件和 `lib_recstore_ops.so` 动态库。
    *   `torch-wheel`: 包含与 RecStore 兼容的 PyTorch 安装包（.whl）。
    *   `recstore-ops`: 包含 RecStore OP 层的动态库（.so）。

如果仓库维护者已经执行过手动 Release 工作流，也可以直接从 **Releases** 页面下载对应版本的 `ps_server`、`recstore_ops`、`ycsb` 和 torch wheel。

## 2. 环境准备

虽然不需要编译环境，但仍需要安装一些基础的运行时依赖。

### 2.1 准备代码仓库

即使是免编译运行，也建议克隆仓库以获取辅助脚本和配置文件：

```bash
git clone https://github.com/RecStore/RecStore.git
cd RecStore
```

### 2.2 安装系统依赖

运行仓库中的初始化脚本安装基础工具（如 `libgoogle-glog-dev` 等运行时库）：

```bash
bash ci/env/init_host_prereqs.sh
```

### 2.3 安装 PyTorch

解压并安装步骤 1 中下载的 `torch-wheel`：

```bash
unzip torch-wheel.zip
pip install binary/torch-*.whl
```

## 3. 部署与运行

### 3.1 解压产物

将 `packed-bundle` 解压并整理到指定目录，模拟构建后的目录结构：

```bash
mkdir -p runner
unzip packed-bundle.zip -d runner

tar -xzf runner/packed-ps-server.tar.gz -C runner
tar -xzf runner/packed-recstore-ops.tar.gz -C runner

mkdir -p build/lib
cp -f runner/package/lib/lib_recstore_ops.so build/lib/
[ -d runner/package/deps/lib ] && cp -f runner/package/deps/lib/*.so* build/lib/
```

### 3.2 运行参数服务器

你可以直接运行 `ps_server`，或者使用我们提供的 `runner.py` 封装脚本（它会自动处理一些库路径和环境配置）：

```bash
export GLOG_logtostderr=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/runner/package/lib:$(pwd)/runner/package/deps/lib

./runner/package/bin/ps_server

# python3 ci/pack/runner.py runner/package bin/ps_server \
#     --ready-pattern "listening on" \
#     --timeout 180 \
#     --keep-alive
```

??? failure "常见错误"

    ```bash
    cp: cannot stat 'runner/package/deps/lib/ld-linux-x86-64.so.2': Too many levels of symbolic links
    ```

    这是因为符号链接形成循环，可以直接删除 `runner/package/deps/lib/ld-linux-x86-64.so.2` 文件。

    ```bash
    od: symbol lookup error: .../runner/package/deps/lib/libc.so.6: undefined symbol: __tunable_is_initialized, version GLIBC_PRIVATE
    ```

    该错误是由于 GLIBC 版本不兼容导致的，可能需要升级 GLIBC 到更高版本。

## 4. Release 包说明

- Release 包会刻意排除宿主机应自行提供的核心运行库，例如 `ld-linux*`、`libc.so*`、`libm.so*`、`libpthread.so*` 等。
- 这样做是为了避免把 GitHub Actions 构建机上的 loader / GLIBC 一起打包，导致目标机器出现更隐蔽的兼容性错误。
- 因此，Release 包仍然要求目标 Linux 环境提供兼容的系统运行时；推荐 Ubuntu 20.04/22.04 或相近发行版。
