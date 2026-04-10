#!/bin/bash
set -euxo pipefail
cd "$(dirname "$0")"

RECSTORE_PATH="/home/xieminhui/jkt/RecStore"
Docker_RECSTORE_PATH="/app/RecStore"

# Persistent volumes for dependency/cache reuse across container recreation
VOLUME_PIP_CACHE="jkt_recstore_pip_cache"
VOLUME_APT_CACHE="jkt_recstore_apt_cache"
VOLUME_APT_LISTS="jkt_recstore_apt_lists"
VOLUME_DLRM_DEPS="jkt_recstore_dlrm_deps"
VOLUME_NVM="jkt_recstore_nvm"
VOLUME_NPM_CACHE="jkt_recstore_npm_cache"
VOLUME_NODE_GYP_CACHE="jkt_recstore_node_gyp_cache"
VOLUME_CODEX_HOME="jkt_recstore_codex_home"
VOLUME_CODEX_CONFIG="jkt_recstore_codex_config"
VOLUME_OPENCODE_HOME="jkt_recstore_opencode_home"

sudo docker run --cap-add=SYS_ADMIN --privileged --security-opt seccomp=unconfined \
--name jkt_recstore --net=host \
-v ${RECSTORE_PATH}:${Docker_RECSTORE_PATH} \
-v ${VOLUME_PIP_CACHE}:/root/.cache/pip \
-v ${VOLUME_APT_CACHE}:/var/cache/apt \
-v ${VOLUME_APT_LISTS}:/var/lib/apt/lists \
-v ${VOLUME_DLRM_DEPS}:/tmp/recstore-dlrm-deps \
-v ${VOLUME_NVM}:/root/.nvm \
-v ${VOLUME_NPM_CACHE}:/root/.npm \
-v ${VOLUME_NODE_GYP_CACHE}:/root/.cache/node-gyp \
-v ${VOLUME_CODEX_HOME}:/root/.codex \
-v ${VOLUME_CODEX_CONFIG}:/root/.config/codex \
-v ${VOLUME_OPENCODE_HOME}:/root/.opencode \
-v /dev/shm:/dev/shm \
-v /dev/hugepages:/dev/hugepages \
-v /dev:/dev -v /nas:/nas \
-w ${Docker_RECSTORE_PATH} --rm -it --gpus all -d recstore
