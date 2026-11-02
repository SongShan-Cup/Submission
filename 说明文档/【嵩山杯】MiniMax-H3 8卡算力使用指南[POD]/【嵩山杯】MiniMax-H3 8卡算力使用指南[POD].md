# 【嵩山杯】MiniMax\-H3 8卡算力使用指南\[POD\]

本文档介绍一套已经验证的完整流程，用于在海光 Pod 上构建

InfiniCore 和 InfiniLM、准备 Python 环境，并通过

`examples/minimax_h3.py` 运行 PyTorch/Diffusers 参考实现。



已验证的源码分支如下：



- InfiniCore：`minimax-h3`

- InfiniLM：`minimax_h3`



**大赛统一输入图片**



![Golde33443\.jpg](图片和附件/Golde33443.jpg)



## 已验证的硬件拓扑



本文档记录的测试使用了**一个计算节点/Pod，共 8 张海光 GPU**，没有使用多个

计算节点，也没有使用跨节点流水线并行。



- 节点/Pod 数量：**1**

- 加速卡总数：**8 张海光 GPU**（`gfx936`，每张 64 GiB）

- 条件编码器部署位置：**GPU 0\-3**（4 张卡）

- H3 Transformer 部署位置：**GPU 4\-7**（4 张卡）

- 编译位置：同一个 Pod

    

用于进入 Kubernetes Pod 的登录节点不属于额外的计算节点。所有模型计算均在

上述单个 8 卡 Pod 内完成。



以下示例假设持久化私有存储在 Pod 内挂载到 `/root/private_data`。如果实际账户

使用其他挂载点，请相应修改 `WORK_HOME`。



## 1\. 进入 Pod



连接登录节点并找到处于运行状态的 Pod：



```Bash
ssh <login-host>

kubectl get pods -n <namespace>
export POD=<pod-name>
export NS=<namespace>
kubectl exec -it "$POD" -n "$NS" -- bash
```



进入 Pod 后执行：



```Bash
export WORK_HOME=/root/private_data
mkdir -p "$WORK_HOME/Projects"
```



建议将代码仓库、模型文件、输入媒体和最终输出都保存在 `$WORK_HOME` 下。

`/workspace` 适合存放编译产物和缓存，但更换 Pod 后其中的内容可能丢失。



## 2\. 初始化海光环境



```Bash
source /opt/dtk/env.sh

export XMAKE_ROOT=y
export INFINI_ROOT=/root/.infini
export PATH="/root/.local/bin:/opt/dtk/cuda/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
```



安装任何软件包之前，先验证厂商提供的 PyTorch：



```Bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("devices:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```



已验证的 Pod 使用 Python 3\.11\.9、PyTorch 2\.9\.0、HIP 6\.3\.26093，以及 8 张

64 GiB 海光 GPU。



不要使用公共 PyPI 上的 PyTorch wheel 替换系统中的厂商版 PyTorch。



## 3\. 将项目克隆到私有存储



请在能够访问 GitHub 的环境中执行以下命令，通常是在登录节点上：



```Bash
mkdir -p ~/Projects

git clone --branch minimax-h3 --recurse-submodules \
    https://github.com/InfiniTensor/InfiniCore.git \
    ~/Projects/InfiniCore

git clone --branch minimax_h3 --recurse-submodules \
    https://github.com/InfiniTensor/InfiniLM.git \
    ~/Projects/InfiniLM
```



确认这些目录在 Pod 中对应为：



```Plain Text
/root/private_data/Projects/InfiniCore
/root/private_data/Projects/InfiniLM
```



如果无法访问 GitHub，可以在其他机器上完成克隆，连同子模块一起打包代码目录，

然后上传到私有存储，并解压到 `$WORK_HOME/Projects`。



为了加快编译，也可以将同一份源码复制到 `/workspace/Projects`，但仍应在

`$WORK_HOME` 下保留持久化副本。



## 4\. 创建独立 Python 环境



创建一个能够访问系统包的虚拟环境：



```Bash
python -m venv --system-site-packages /workspace/envs/minimax-h3
source /workspace/envs/minimax-h3/bin/activate
```



`--system-site-packages` 非常重要，因为可用的海光/DTK PyTorch 是由容器提供并

安装在系统环境中的。普通的隔离 Conda 环境会隐藏它，除非再次安装完全匹配的

厂商版 PyTorch。



在该虚拟环境中安装 Python 依赖。已验证的版本如下：



```Plain Text
accelerate==1.14.0
diffusers==0.40.0.dev0
transformers==5.14.1
imageio==2.37.4
huggingface_hub==1.28.0
safetensors==0.8.0
regex==2026.5.9
janus==2.0.0
xxhash==4.0.2
av==18.1.0
einops==0.8.2
pytest==9.1.1
flash-attn==2.6.1+das.opt1.dtk2604.torch290
triton==3.5.1+das.opt1.dtk2604.torch290
```



Diffusers 必须包含 `diffusers.modular_pipelines.minimax_h3`。请使用与已验证

NVIDIA 环境相同的 Diffusers 源码版本或软件包。



能够访问软件包索引时，执行：



```Bash
python -m pip install \
    accelerate==1.14.0 \
    transformers==5.14.1 \
    imageio==2.37.4 \
    huggingface_hub==1.28.0 \
    safetensors==0.8.0 \
    regex==2026.5.9 \
    janus==2.0.0 \
    xxhash==4.0.2 \
    av==18.1.0 \
    einops==0.8.2 \
    pytest==9.1.1
```



单独安装兼容的 Diffusers 开发版源码：



```Bash
python -m pip install -e /path/to/compatible/diffusers \
    --no-deps --no-build-isolation
```



对于离线 Pod，可在其他机器上下载 CPython 3\.11 Linux wheel，将其上传到

`$WORK_HOME/wheels`，然后在不访问软件包索引的情况下安装：



```Bash
python -m pip install --no-index --find-links "$WORK_HOME/wheels" \
    accelerate transformers imageio huggingface_hub safetensors regex \
    janus xxhash av
```



## 5\. 针对海光 PyTorch 构建 torchvision



通用 Linux torchvision wheel 无法与厂商版 PyTorch 正常配合，原因是

`torchvision::nms` 没有注册。因此需要使用当前海光 PyTorch 从源码构建

torchvision 0\.24\.0。



```Bash
source /opt/dtk/env.sh
source /workspace/envs/minimax-h3/bin/activate

git clone --depth 1 --branch v0.24.0 \
    https://github.com/pytorch/vision.git \
    /workspace/third_party/vision

cd /workspace/third_party/vision
export BUILD_VERSION=0.24.0
export MAX_JOBS=16
export FORCE_CUDA=1
python -m pip install -v . --no-build-isolation --no-deps
```



离开 torchvision 源码目录后进行验证：



```Bash
cd /workspace
python - <<'PY'
import torch
import torchvision
from torchvision.transforms import functional

print("torch:", torch.__version__, torch.version.hip)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("torchvision::nms:", hasattr(torch.ops.torchvision, "nms"))
print("transforms:", functional.__name__)
PY
```



最后两项检查必须成功。如果缺少 torchvision，Qwen3\-VL processor 无法创建，

后续生成会出现：



```Plain Text
AttributeError: 'NoneType' object has no attribute 'image_processor'
```



## 6\. 配置并构建 InfiniCore



已验证的 Pod 使用 `gfx936`。只有在目标设备报告其他架构时才应修改该值。



```Bash
source /opt/dtk/env.sh
source /workspace/envs/minimax-h3/bin/activate

export XMAKE_ROOT=y
export INFINI_ROOT=/root/.infini
export PATH="/root/.local/bin:/opt/dtk/cuda/cuda/bin:${PATH}"

cd /root/private_data/Projects/InfiniCore
xmake global --network=private
xmake f -y -cv \
    --hygon-dcu=y \
    --hygon-arch=gfx936 \
    --cuda=/opt/dtk/cuda/cuda \
    --ccl=y \
    --aten=y

xmake build -j 16
xmake install
xmake build -j 16 _infinicore
xmake install _infinicore
python -m pip install -e . --no-deps --no-build-isolation
```



验证 Python 包：



```Bash
python -c 'import infinicore; print(infinicore.__file__)'
```



## 7\. 构建 InfiniLM



```Bash
source /opt/dtk/env.sh
source /workspace/envs/minimax-h3/bin/activate

export XMAKE_ROOT=y
export INFINI_ROOT=/root/.infini
export PATH="/root/.local/bin:/opt/dtk/cuda/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"

cd /root/private_data/Projects/InfiniLM
xmake build -j 16 infinicore_infer
xmake install infinicore_infer
xmake build -j 16 _infinilm
xmake install _infinilm
python -m pip install -e . --no-deps --no-build-isolation
```



验证两个软件包：



```Bash
python - <<'PY'
import infinicore
import infinilm

print("infinicore:", infinicore.__file__)
print("infinilm:", infinilm.__file__)
PY
```



## 8\. 准备模型文件和输入媒体



已验证的模型目录结构如下：



```Plain Text
/root/private_data/shared/MiniMax-H3/
|-- diffusers/
|   |-- modular_model_index.json
|   |-- processor/
|   |-- text_encoder/
|   |-- transformer/
|   |-- vae/
|   `-- audio_vae/
`-- Ref2VA/
    |-- text_encoder/
    `-- transformer/
```



将输入图片放入持久化存储，例如：



```Plain Text
/root/private_data/Golde33443.jpg
```



创建输出目录：



```Bash
mkdir -p /root/private_data/minimax_h3_reference_test
```



## 9\. 配置注意力后端



海光 PyTorch 会动态加载名为 `flash_attn_2_cuda*.so` 的 FlashAttention

动态库。必须安装同时匹配以下三部分运行时 ABI 的插件：Python 3\.11、DTK 26\.04

和 PyTorch 2\.9\.0。



```Bash
source /opt/dtk/env.sh
source /workspace/envs/minimax-h3/bin/activate

# Pod 预设的代理可能不可用。可以直接访问厂商软件包索引，因此在此命令中
# 同时移除大小写形式的代理环境变量。
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    python -m pip install --no-deps \
    "triton==3.5.1+das.opt1.dtk2604.torch290" \
    "flash-attn==2.6.1+das.opt1.dtk2604.torch290" \
    --index-url https://pypi.sourcefind.cn/release/dtk/
```



PyTorch 本身安装在容器的系统 site\-packages 中，而插件安装在虚拟环境内，因此

需要显式告知 PyTorch dispatcher 扩展库的位置：



```Bash
export FA_SO_PATH="$(find "$VIRTUAL_ENV"/lib/python*/site-packages \
    -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit)"
test -n "$FA_SO_PATH"
```



每次在新的 shell 中启动 MiniMax\-H3 时都要设置 `FA_SO_PATH`。也可以在一次性

Pod 中以 root 身份创建自动发现符号链接：



```Bash
ln -sfn "$FA_SO_PATH" \
    "/usr/local/lib/python3.11/site-packages/$(basename "$FA_SO_PATH")"
```



加载完整模型之前，先验证注意力后端：



```Bash
python - <<'PY'
import torch
import torch.nn.functional as F

q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
out = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
print(out.shape, out.dtype, out.device)
PY
```



已验证的运行会打印 `sdpa adopt the new interface of flash-attn`。与 math SDPA

进行数值对比时，最大绝对误差为 `0.0078125`，平均绝对误差为 `0.0006035`，

并且以 `rtol=0.02, atol=0.02` 通过 `torch.allclose` 检查。



匹配的 Triton 软件包加上 `einops` 和 `pytest` 后，也可以直接使用软件包的

Python API。已使用 `from flash_attn import flash_attn_func` 和形状为

`[1, 128, 8, 64]` 的 BF16 张量完成验证。



## 10\. 运行已验证的 FL2VA 参考测试



以下命令启用 FlashAttention，并使用单个节点上的全部 8 张卡：其中 4 张用于

条件编码器，另外 4 张用于 H3 Transformer。



```Bash
source /opt/dtk/env.sh
source /workspace/envs/minimax-h3/bin/activate

export INFINI_ROOT=/root/.infini
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
export FA_SO_PATH="$(find "$VIRTUAL_ENV"/lib/python*/site-packages -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit)"

cd /root/private_data/Projects/InfiniLM
python examples/minimax_h3.py fl2va \
    --model-path=/root/private_data/shared/MiniMax-H3/diffusers \
    --text="A dog falls asleep" \
    --first-frame=/root/private_data/Golde33443.jpg \
    --gpus=0,1,2,3,4,5,6,7 \
    --legacy \
    --legacy-text-gpus=0,1,2,3 \
    --legacy-transformer-gpus=4,5,6,7 \
    --num-frames=124 \
    --steps=10 \
    --seed=42 \
    --height=384 \
    --width=576 \
    --output=/root/private_data/minimax_h3_reference_test/fl2va_legacy.mp4
```



已验证输出的媒体信息如下：



```Plain Text
分辨率：576x384
帧数：  124
帧率：  24 FPS
时长：  5.17 秒
音频：  存在
```



FlashAttention 路径使用两个去噪步骤完成了端到端验证，测得时间如下：



```Plain Text
模型加载                           273.71 秒
条件编码器计算                       6.97 秒
H3 Transformer 计算                  8.95 秒
视频和音频解码                        7.94 秒
生成流水线总计                       25.55 秒
端到端总计                         300.28 秒
```



模型加载时间异常偏大是由 checkpoint 存储 I/O 引起的，与 FlashAttention 无关。

快速功能冒烟测试可使用 `--steps=2`；进行质量评估时，请恢复为 `--steps=10` 或

所需的正式步数。



## 11\. 使用 FlashAttention 运行 Ref2VA



使用 FL2VA 的输出作为 Ref2VA 输入：



```Bash
cd /root/private_data/Projects/InfiniLM
export FA_SO_PATH="$(find "$VIRTUAL_ENV"/lib/python*/site-packages -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit)"
python examples/minimax_h3.py ref2va \
    --model-path=/root/private_data/shared/MiniMax-H3/diffusers \
    --text="Turn the video into cartoon style" \
    --reference-video=/root/private_data/minimax_h3_reference_test/fl2va_legacy.mp4 \
    --gpus=0,1,2,3,4,5,6,7 \
    --legacy \
    --legacy-text-gpus=0,1,2,3 \
    --legacy-transformer-gpus=4,5,6,7 \
    --num-frames=124 \
    --steps=10 \
    --seed=42 \
    --height=384 \
    --width=576 \
    --output=/root/private_data/minimax_h3_reference_test/ref2va_legacy_cartoon.mp4
```



该路径现在可以使用厂商版 FlashAttention 插件完整运行。一次包含 124 帧、

384x576 分辨率、两个去噪步骤的验证运行已成功完成，没有再出现此前的 OOM，

并写入：



```Plain Text
/root/private_data/minimax_h3_flash_test/ref2va_flash_2step.mp4
```



## 12\. 故障排查摘要



### `xmake` 拒绝以 root 身份运行



```Plain Text
Running xmake as root is extremely dangerous
```



请设置：



```Bash
export XMAKE_ROOT=y
```



### 缺少 `janus` 或 `xxhash`



即使选择 MiniMax legacy 路径，InfiniLM 也会提前导入其 LLM 软件包。请在隔离的

虚拟环境中安装这两个依赖。



### processor 为 `None`



检查更早的日志中是否存在被抑制的 processor 加载错误。在已验证的 Pod 上，

原因是缺少 torchvision。请按照前文说明，针对海光 PyTorch 从源码构建

torchvision。



### `No matching libraries found for flash_attn_2_cuda*.so`



海光 PyTorch SDPA dispatcher 选择了一个不存在的 FlashAttention 插件。请安装

第 9 节列出的精确厂商版本，并设置 `FA_SO_PATH`。不要安装公共 CUDA wheel，

因为它与海光 PyTorch ABI 不匹配。



### `scaled_dot_product_attention` 报 `HIP out of memory`



Math SDPA 会物化一个很大的注意力矩阵。全分辨率 FL2VA 曾尝试分配约

158\.57 GiB，而完整 Ref2VA 曾尝试分配约 442\.14 GiB。应使用厂商版

FlashAttention 后端。正常运行 MiniMax\-H3 时，不要再使用旧的 math\-SDPA

启动器。



### `encode_video` 需要 PyAV



安装兼容的 Linux wheel：



```Bash
python -m pip install av==18.1.0
```



对于离线 Pod：



```Bash
python -m pip install --no-index "$WORK_HOME/wheels/av-18.1.0-"*.whl
```



