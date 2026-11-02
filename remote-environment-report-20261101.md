# 远程 H3 环境检查（2026-11-01）

目标 Pod：`vcjob-2116206198177124354-aclsylqidf-33085-worker-0`，命名空间 `aclsylqidf`。
本次仅上传源码、检查依赖；未安装软件、修改环境变量配置或编译运行 H3。

## 上传范围

可直接使用的源码目录：`/workspace/minimax-h3-upload-20261101`。
持久化备份目录：`/root/private_data/minimax-h3-upload-20261101`。
包含 InfiniCore、InfiniLM、MiniMax-H3、现有子模块源码、offline_h3、说明文档、两个根目录 Python 示例及要求.txt；附带 remote-install 的安装脚本、验证脚本和说明/版本清单。
排除 Git 历史、Python 缓存、下载虚拟环境、大型离线依赖包和重复压缩包。

已完成上传：7,748 个源码/配置/文档文件，共 216,542,076 字节；压缩包 119,206,831 字节。远程压缩包和解包后全部文件均通过 SHA-256 校验；两处目录均有 `UPLOAD_VERIFIED.json`，源码目录另有 `UPLOAD_MANIFEST.json`。

压缩包 SHA-256：`dee67d99c8966bb6a47754a3f5db06e82907014e57c5edb956c227471a000527`。

上传时首次 SCP 握手被远端关闭，重试成功。持久化目录解包小文件明显缓慢，已报告后停止，将未完成的文件保留在备份目录的 `partial-extraction` 下；完整源码在 `/workspace` 解包并校验成功。`/workspace` 随容器生命周期可能丢失，持久化 `workspace.tar.gz` 和清单可用于恢复。

## 已确认可用

- Python 3.11.9，`/usr/local/bin/python`；Linux x86_64，glibc 2.35。
- 厂商 PyTorch 发行版 `2.9.0+das.opt1.dtk2604`，模块版本 2.9.0，HIP 6.3.26093，C++11 ABI 为 True。
- `torch.cuda.is_available()` 为 True，识别 8 张 BW 卡，架构 `gfx936:sramecc+:xnack-`，单卡显存 68,702,699,520 字节（约 64 GiB）。未运行计算负载。
- GCC 11.4.0、Clang 17、hipcc、CMake 可用。
- DTK 的 CUDA 兼容工具位于 `/opt/dtk/cuda/cuda`（实际为 `/opt/dtk-26.04/cuda/cuda-12`），其中有 `bin/nvcc`、`include/nccl.h` 和 `lib64/libnccl.so`。仅确认文件存在，未做编译链接验证。
- NumPy 1.25.0、Pillow 12.2.0、Transformers 4.57.6、safetensors 0.7.0、huggingface_hub 0.36.0 导入成功；`Qwen3VLForConditionalGeneration` 导入成功。
- 模型目录 `/root/private_data/shared/MiniMax-H3/diffusers` 存在；model_index.json、modular_model_index.json、transformer、text_encoder、vae、audio_vae 存在。3 个 safetensors 索引引用的 31 个不同分片文件均存在。此检查不验证权重内容、完整性或数值正确性。

## 已报告的问题

1. 当前系统 Python 未安装 Diffusers、Accelerate、InfiniCore、InfiniLM、flash-attn、triton、ninetoothed、ntops、soundfile、imageio、imageio-ffmpeg、av、janus、xxhash、scikit-image、einops、pybind11。
2. PATH 中没有 xmake，也没有 ffmpeg 命令；后续需验证媒体编码支持。
3. `CUDA_HOME=/usr/local/cuda` 指向不存在的目录。DTK 兼容 nvcc 已存在，但不在当前 PATH 中。`INFINI_ROOT` 和 `FA_SO_PATH` 未设置，`/root/.infini` 不存在。
4. `python -m pip check` 报告：pandas 3.0.2 要求 NumPy >=1.26.0，当前为 1.25.0。
5. 本地 remote-install 安装说明明确表示离线包不构成完整环境：仍需 H3 Diffusers、匹配的 torchvision、InfiniCore/InfiniLM 原生构建及九齿开发环境；其 FlashAttention wheel 不含源码构建配置引用的 `libflash_attention.so`。是否采用该构建路径需后续核对。
6. 当前工作区根目录不是可用 Git 仓库，上传以实际文件快照为准。各项目的 Git 历史不随本次源码快照上传。

## 后续处理建议（尚未执行）

先评估远程已有的 H3 依赖归档和构建缓存是否匹配当前源码及 DTK/PyTorch ABI，再在独立环境中补齐依赖，保留厂商 PyTorch。H3 入口直接导入 `MiniMaxH3Transformer3DModel`、`ModularPipeline` 及 H3 modular pipeline API，不能只按 Diffusers 版本号判断可用。

确认构建链、FlashAttention 链接依赖及包版本兼容后，再构建 InfiniCore/InfiniLM，执行导入、小张量及最终推理验证。当前不能宣称 H3 已可运行。
