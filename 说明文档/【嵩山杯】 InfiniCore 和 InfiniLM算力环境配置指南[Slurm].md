# 【嵩山杯】 InfiniCore 和 InfiniLM算力环境配置指南\[Slurm\]

**请各位选手及队伍注意：**

初赛起始阶段，所有参赛队伍仅分配单卡资源，用于完成 MiniMax H3 模型的算子层优化。算子优化完成后，请撰写一份简要技术报告，简要包含队伍信息、被优化算子及基线环境、优化思路、性能与正确性对比、代码位置和复现方式，并提交至 **duhang@qiyuanlab\.com**。技术报告不设提交截止时间，但**只有提交了技术报告的队伍，才会获得 8 卡资源账号**。获得 8 卡资源后，需在初赛结束前完成 8 卡的模型适配。



本文记录在 SCNet 海光队列上从零配置 InfiniCore/InfiniLM，并通过 `srun` 完成构建、算子测试和小模型推理测试的步骤。不要在登录节点直接进行编译、算子测试或模型推理。



## 1\. 已验证环境



本次实测环境如下：



- Slurm 分区：`hx1hdnormal`

- 资源：1 个节点、1 个 task、16 个 CPU 核、1 张 DCU

- 海光架构：`gfx936`，PyTorch 中设备名为 `BW`

- DTK module：`compiler/dtk/26.04`

- SGHPC module：`sghpcdas/25.6`

- Python：3\.10\.19

- PyTorch：2\.5\.1，HIP 6\.3\.25405

- InfiniCore 分支：`minimax-h3`

- InfiniLM 分支：`minimax_h3`

- 小模型：`Qwen/Qwen2.5-0.5B-Instruct`

    

不同账号和计算中心的软件路径可能不同。先运行环境探测命令，再按实际结果调整 module 和 Python 路径。



## 2\. 确认 Slurm 和 DCU



登录 SCNet HPC E\-Shell 后检查队列：



```Bash
whichpartition
sinfo -o '%P %a %l %D %G'
```



该队列要求作业显式申请 DCU。以下是后续统一使用的 `srun` 规格：



```Bash
srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gres=dcu:1 \
  --time=00:10:00 \
  bash -lc 'hostname; hy-smi'
```



注意：



- `--ntasks=1` 表示只启动一个进程；不要误配成 16 个 task。

- `srun` 是同步命令，命令结束后资源会自动释放。

- SSH 断开会影响前台 `srun`；长时间无人值守任务可把相同资源参数改写成 `sbatch` 脚本。

- 可用 `squeue -u "$USER"` 和 `scancel <job-id>` 查看、终止作业。

    

## 3\. 准备代码



在登录节点创建持久化目录：



```Bash
mkdir -p ~/Projects ~/.infini ~/venvs ~/models
cd ~/Projects
```



克隆目标分支：



```Bash
git clone --branch minimax-h3 --recurse-submodules \
  https://github.com/InfiniTensor/InfiniCore.git

git clone --branch minimax_h3 --recurse-submodules \
  https://github.com/InfiniTensor/InfiniLM.git
```



已有仓库时：



```Bash
cd ~/Projects/InfiniCore
git fetch origin
git checkout minimax-h3
git pull --ff-only origin minimax-h3
git submodule update --init --recursive

cd ~/Projects/InfiniLM
git fetch origin
git checkout minimax_h3
git pull --ff-only origin minimax_h3
git submodule update --init --recursive
```



## 4\. 准备 Python 和 xmake



本环境可复用平台提供的海光 PyTorch，避免安装普通 CUDA 版 PyTorch：



```Bash
module purge
module load compiler/dtk/26.04
module load sghpcdas/25.6

PLATFORM_PYTHON=/public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/envs/vllm-25.04.2/bin/python
"$PLATFORM_PYTHON" -m venv --system-site-packages ~/venvs/infinilm
source ~/venvs/infinilm/bin/activate

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('HIP:', torch.version.hip)
print('DCU count:', torch.cuda.device_count())
PY
```



如果 `xmake` 尚未安装，可在允许联网的登录环境安装到用户目录：



```Bash
curl -fsSL https://xmake.io/shget.text | bash
export PATH="$HOME/.local/bin:$PATH"
xmake --version
```



实测使用 xmake 3\.1\.2。



## 5\. 创建统一环境脚本



保存为 `~/setup_env.sh`：



```Bash
cat > ~/setup_env.sh <<'EOF'
#!/usr/bin/env bash

module purge
module load compiler/dtk/26.04
module load sghpcdas/25.6
source "$HOME/venvs/infinilm/bin/activate"

if [[ -f "$HOME/.xmake/profile" ]]; then
    restore_nounset=0
    if [[ $- == *u* ]]; then
        restore_nounset=1
        set +u
    fi
    source "$HOME/.xmake/profile"
    if [[ $restore_nounset -eq 1 ]]; then
        set -u
    fi
fi

export XMAKE_ROOT=y
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini}"
export DTK_ROOT="${DTK_ROOT:-$DTKROOT}"
export HYGON_CUDA_HOME="${HYGON_CUDA_HOME:-$DTK_ROOT/cuda/cuda}"
export PYTHON="${PYTHON:-$HOME/venvs/infinilm/bin/python}"

torch_lib=$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)
python_lib=$(python - <<'PY'
import sys
print(sys.base_prefix + '/lib')
PY
)

export PATH="$HOME/.local/bin:$HYGON_CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$DTK_ROOT/lib:$HYGON_CUDA_HOME/lib64:$python_lib:$torch_lib:${LD_LIBRARY_PATH:-}"
EOF

chmod +x ~/setup_env.sh
```



## 6\. 构建、安装 InfiniCore



创建 `~/build_infinicore.sh`：



```Bash
cat > ~/build_infinicore.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/setup_env.sh"
cd "$HOME/Projects/InfiniCore"

xmake f -y -m release \
  --hygon-dcu=y \
  --hygon-arch=gfx936 \
  --cuda="$HYGON_CUDA_HOME" \
  --aten=n \
  --ccl=y

xmake build -j "${SLURM_CPUS_PER_TASK:-16}"
xmake install
xmake build -j "${SLURM_CPUS_PER_TASK:-16}" _infinicore
xmake install _infinicore
python -m pip install -e . --no-deps --no-build-isolation
EOF
chmod +x ~/build_infinicore.sh
```



通过 `srun` 构建：



```Bash
srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gres=dcu:1 \
  --time=04:00:00 \
  bash ~/build_infinicore.sh
```



这里使用 `--aten=n`。实测平台默认 GCC 低于 PyTorch C\+\+ 头文件要求的 GCC 9，启用可选 ATen adaptor 会在包含 `c10/util/C++17.h` 时失败。InfiniLM 原生推理和本指南中的 InfiniCore 算子测试不依赖该 adaptor。



## 7\. 运行 InfiniCore 算子测试



```Bash
srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gres=dcu:1 \
  --time=00:20:00 \
  bash -lc 'source ~/setup_env.sh && cd ~/Projects/InfiniCore && python test/infinicore/ops/timestep_embedding.py --hygon'
```



实测结果：18 个测试全部通过，成功率 100%。测试覆盖 FP16、BF16、FP32，以及多种输入长度和 embedding 维度。



## 8\. 构建、安装 InfiniLM



当前分支的 xmake 工程只有 `_infinilm` 目标，不再包含旧的 `infinicore_infer` 目标：



```Bash
cat > ~/build_infinilm.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/setup_env.sh"
cd "$HOME/Projects/InfiniLM"

xmake f -y -m release
xmake build -j "${SLURM_CPUS_PER_TASK:-16}" _infinilm
xmake install _infinilm
python -m pip install -e . --no-deps --no-build-isolation
EOF
chmod +x ~/build_infinilm.sh

srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gres=dcu:1 \
  --time=04:00:00 \
  bash ~/build_infinilm.sh
```



## 9\. 安装 Python 运行时依赖



平台 vLLM 环境已提供 PyTorch、Transformers、Safetensors、Tokenizers 和 Hugging Face Hub。本次额外缺少 `janus`：



```Bash
srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --gres=dcu:1 \
  --time=00:20:00 \
  bash -lc 'source ~/setup_env.sh && python -m pip install janus'
```



验证导入：



```Bash
srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --gres=dcu:1 \
  --time=00:10:00 \
  bash -lc 'source ~/setup_env.sh && python - <<"PY"
import infinicore
import infinilm
import torch
print("infinicore:", infinicore.__file__)
print("infinilm:", infinilm.__file__)
print("torch:", torch.__version__, "HIP:", torch.version.hip)
PY'
```



如果平台环境还缺少其他依赖，再安装：



```Bash
python -m pip install janus xxhash huggingface_hub transformers safetensors tokenizers
```



不要覆盖平台提供的海光 PyTorch。



## 10\. 下载小模型



本次使用 `Qwen/Qwen2.5-0.5B-Instruct`：



```Bash
cat > ~/download_smoke_model.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/setup_env.sh"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

target = os.path.expanduser('~/models/Qwen2.5-0.5B-Instruct')
snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir=target)
print('model:', target)
PY
EOF
chmod +x ~/download_smoke_model.sh

srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gres=dcu:1 \
  --time=01:00:00 \
  bash ~/download_smoke_model.sh
```



模型目录至少应包含 `config.json`、`model.safetensors`、`tokenizer.json` 和 `tokenizer_config.json`。



## 11\. 运行 InfiniLM 小模型冒烟测试



```Bash
cat > ~/run_infinilm_smoke.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$HOME/setup_env.sh"
cd "$HOME/Projects/InfiniLM"

python examples/test_infer.py \
  --device=hygon \
  --model="$HOME/models/Qwen2.5-0.5B-Instruct" \
  --enable-paged-attn \
  --attn=paged-attn \
  --num-blocks=8 \
  --max-new-tokens=16 \
  --top-k=1 \
  --temperature=1.0 \
  --prompt='Explain why the sky is blue in one sentence.'
EOF
chmod +x ~/run_infinilm_smoke.sh

srun \
  --partition=hx1hdnormal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gres=dcu:1 \
  --time=00:30:00 \
  bash ~/run_infinilm_smoke.sh
```



实测成功输出：



```Plain Text
The sky is blue because of the scattering of sunlight by tiny particles in the Earth
```



本次权重加载约 382 ms，生成阶段总耗时约 385 ms。



## 12\. 可选：直接联网失败时使用代理



只有在 GitHub、PyPI 或 Hugging Face 直连失败时才需要代理。端口和登录地址必须由当前用户自行选择，不要照抄固定值。



本地开发机已有 HTTP 代理时，可建立反向 SSH 转发：



```Bash
ssh -N \
  -R <remote-reverse-port>:127.0.0.1:<local-proxy-port> \
  <login-host>
```



登录节点自身可使用：



```Bash
export HTTP_PROXY=http://127.0.0.1:<remote-reverse-port>
export HTTPS_PROXY="$HTTP_PROXY"
```



计算节点不能直接访问登录节点的 `127.0.0.1`。如果下载必须在 `srun` 内执行，需要在登录节点临时启动 TCP relay，将一个集群内可访问的端口转发到反向 SSH 端口，然后把登录节点主机名和 relay 端口作为 `HTTP_PROXY`/`HTTPS_PROXY` 传给作业。下载完成后立即停止 relay 和 SSH 转发，避免长期暴露代理。



如果不希望维护 relay，更简单的做法是在登录节点下载 wheel/模型到共享的 `$HOME`，随后仅在 `srun` 作业中执行安装和测试。



## 13\. 故障排查



### `No partition specified` 或没有可用分区



确认当前 SSH/E\-Shell 账号属于购买资源的计算中心，并执行 `whichpartition`。Notebook/Pod 账号与 HPC Slurm 账号可能不是同一个账号。



### QOS 要求申请 DCU



保留 `--gres=dcu:1`。



### 输出重复多次



检查是否误用了多个 task。单进程构建和推理应使用 `--ntasks=1 --cpus-per-task=16`。



### PyTorch 报 GCC 版本过低



如果错误来自 `c10/util/C++17.h`，说明启用了 ATen adaptor，但系统 GCC 不满足 PyTorch C\+\+ 头文件要求。本指南使用 `--aten=n`。



### `ModuleNotFoundError: janus`



在 `~/venvs/infinilm` 中执行 `python -m pip install janus`。



### pip 或 Hugging Face 无法解析域名



计算节点默认可能没有外网 DNS。使用第 12 节的可选代理，或在登录节点准备好 wheel/模型后再运行计算作业。



### 查看和释放资源



```Bash
squeue -u "$USER"
scontrol show job <job-id>
scancel <job-id>
```



## 14\. 官方参考



- [命令行提交作业](https://www.scnet.cn/help/docs/mainsite/hpc/cmd/cmd-submit-job/)

- [Slurm 作业调度系统](https://www.scnet.cn/help/docs/mainsite/hpc/cmd/slurm/)

