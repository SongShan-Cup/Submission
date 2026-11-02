# MiniMax-H3 接入检查与服务器验收

检查日期：2026-10-27。检查基线：InfiniLM `65deb91`、MiniMax-H3 `d21241f`；同时检查了本地 InfiniCore `f32e2fbb` 的相关算子。

本轮完成的是代码逻辑对齐。本地未编译 C++ 扩展、未加载模型权重、未运行 GPU 推理或数值精度测试。以下“已修改”不等同于“服务器验收通过”。

## 接入现状

| 模块 | 当前状态 |
| --- | --- |
| H3 DiT | 已有原生实现：视频/音频输入投影、文本 refiner、50 层 transformer、AdaLN、三轴 RoPE、双输出头及张量并行 |
| Qwen3-VL 条件编码器 | 已有原生实现：视觉编码、DeepStack、前 50 层文本网络、跳过最终 norm，输出 `hidden_states[50]` |
| 权重加载 | 支持原始 checkpoint 与 Diffusers 名称映射；处理按 head 交错的 QKV、SwiGLU 及混合精度模块 |
| Pipeline | 由 Diffusers 完成输入组织、参考素材处理、调度、视频/音频编解码 |
| 验收 | 原有脚本比较视频尺寸、帧数及灰度 SSIM；不能替代模型内部或音频验收 |

## 本轮已修改

### 1. 视频条件编码缺少时间片之间的注意力隔离

位置：`csrc/models/qwen3_5/qwen3_5_vision.cpp` / `.hpp`。Qwen3-VL 复用这个视觉模型。

之前 Python 将一个视频的全部 patch 传给视觉网络，C++ 将 `t*h*w` 个 token 放在一个 attention 序列中，导致不同时间片之间互相注意。参考 Qwen3-VL 通过 `cu_seqlens` 将每个时间片划为独立序列。

现将每个时间片的 `h*w` 个 token 作为独立 MHA batch，保持展平后的输出顺序以及 DeepStack/merger 的排列。单图仍对应 batch=1。新增 grid 与像素行数检查。

这个文件也被 Qwen3.5 复用，服务器需要覆盖它的单图与视频输入，检查共享实现的回归。

### 2. Ref2VA 的 PyTorch 参考权重转换遗漏 SwiGLU 重排

位置：`examples/minimax_h3.py::_remap_legacy_h3_tensor`。

原始 `mlp.fc1.weight` 为 `[gate, value]`；Diffusers 为 `[value, gate]`。原生加载器已区分两者，而旧参考加载器仅重命名，因此两个后端实际使用了不同的 FFN。现同时修复普通 block 和 token refiner 的权重交换。

原说明文档中的 Ref2VA 5.523% SSIM 不能直接归因于 InfiniLM 数值误差：参考加载器与原生视频注意力都存在结构问题。需在服务器重新生成两端结果。

### 3. 模型路径与当前仓库布局不一致

位置：`examples/minimax_h3.py::resolve_model_paths`、`load_pipeline`。

现支持：

- 当前布局：根目录 `transformer/` 用于 FL2VA，`transformer_ref/` 用于 Ref2VA，共享根目录 `text_encoder/`。
- 旧布局：根目录下另有 `diffusers/`，缺少转换后的 Ref2VA 组件时回退到 `Ref2VA/transformer/`。
- 显式指定 `FL2VA/` 或 `Ref2VA/`：选用该分区的原始权重，仍从根目录或 `diffusers/` 加载 modular pipeline。

PyTorch 参考加载方式现在由 transformer 的 `_class_name` 决定，不再把 FL2VA 固定视为转换格式、Ref2VA 固定视为原始格式。

### 4. 原生权重加载的兼容性缺口

位置：`python/infinilm/modeling_utils.py`。

- 支持 H3 的 `diffusion_pytorch_model.safetensors.index.json`，按索引选择分片，避免退化为扫描目录中的所有 safetensors。
- 原始格式识别覆盖仅包含 `token_refiner.blocks.*` 或 `time_embedder.proj_*` 的分片，避免因同一分片没有 `blocks.*` 等主干参数而漏做转换。

### 5. AdaLN 运算顺序与参考实现不同

位置：`csrc/models/minimax_h3/minimax_h3_transformer.cpp::apply_adaln`。

按参考实现改为 `x * (1 + scale) + shift`，替换原先的 `x + x * scale + shift`。这是代码顺序对齐，尚未量化其对输出的影响；底层算子的舍入、融合和累加精度仍待服务器验证。

### 6. Python bridge 的设备和配置一致性

位置：`python/infinilm/multimodal/minimax_h3.py`。

- 未显式传 `tp_device_ids` 时，从 `device_id` 起选择连续设备，避免 Python 输入在 GPU N、原生引擎却默认从 GPU 0 开始。
- Torch 设备类型使用相同的设备映射，不再固定为 CUDA。
- RoPE 同时识别原始配置的 `rope_inv_freq_len` 与转换配置的 `rope_freq_dim`，与 C++ 选取顺序一致。

设备映射修复并不代表已经实现所有硬件后端，见下面的限制。

### 7. 只比较已有视频时仍强制解析模型目录

位置：`examples/minimax_h3_compare.py::main`。

原先可选的 `--model-path` 未传时，会先执行 `Path(None)`，连只比较已有视频也会报错。现仅在需要生成结果时要求模型路径，并给出明确提示。

## 仍未完成或需要另行决定的适配

1. **完整 InfiniCore 化未完成。** 视频 VAE、音频 VAE、调度器、预处理、RoPE cache 构建仍由 Python/PyTorch/Diffusers 执行。如果目标是替换 DiT 和条件编码器，目前架构符合这个范围；如果要求全链路原生化，还需继续移植这些部分。
2. **非 NVIDIA 平台未完成。** 示例显式要求 CUDA；本地 `timestep_embedding/operator.cc` 仅分发 NVIDIA/HYGON，没有 Ascend 等实现。修改设备名称不足以在昇腾等平台运行，需逐项补齐实际执行路径上的算子和媒体组件。
3. **批处理未完成。** 原生 bridge 限制 batch=1，条件编码器拒绝 padding。当前适用于单请求 packed sequence；多请求批处理还需文档边界、索引和 attention mask 支持。
4. **音频与内部状态验收未完成。** 当前 compare 脚本只检查视频；需在服务器另行检查音频输出、条件 embedding、单步双模态输出及 TP 一致性。
5. **依赖版本未锁定。** 现有环境说明没有冻结可复现的 Diffusers/Transformers 版本。服务器应记录成功运行版本或 commit，尤其要确认 H3 modular pipeline 和 Qwen3-VL processor API 可用。
6. **性能优化未完成。** Python/native 交界每步复制输出并同步设备；长视频没有专门的序列并行或 H3 图编译接入。本轮不把这些优化作为正确性修复的一部分。

## 服务器验收顺序

先使用本地对应分支的 InfiniCore 编译安装，再重新编译 InfiniLM。此次修改包含 C++，仅更新 Python 文件不够。

以下命令在 InfiniLM 仓库目录运行。替换模型与输入路径、GPU 列表。`--model-path` 可直接指向新版模型根目录；模型权重需要在服务器实际存在。

```bash
xmake f -m release -cv
xmake build _infinilm
xmake install _infinilm
python -m pip install -e .

# 先做少步数的运行检查；它不代表生成质量验收。
python examples/minimax_h3.py fl2va \
  --model-path=/models/MiniMax-H3 \
  --text="A dog falls asleep" --first-frame=/inputs/first.jpg \
  --gpus=0,1,2,3 --num-frames=124 --steps=2 \
  --output=smoke_fl2va.mp4

# 使用真实多帧参考视频，以覆盖这次修改的视觉 attention 路径。
python examples/minimax_h3.py ref2va \
  --model-path=/models/MiniMax-H3 \
  --text="Transform the reference video into a colorful hand-drawn cartoon animation style" \
  --reference-video=/inputs/reference.mp4 \
  --gpus=0,1,2,3 --num-frames=124 --steps=2 \
  --output=smoke_ref2va.mp4
```

随后分别覆盖转换后的 `transformer_ref/` 与显式原始 `Ref2VA/` 路径，两种后端都应能加载。另覆盖首尾帧、单图参考、视频参考、视觉参考加音频，以及不从 GPU 0 开始的设备列表。

等运行检查通过后，再在服务器开展精度比较；先修正后的参考基线，再比较完整生成结果。视频对比命令示例：

```bash
python examples/minimax_h3_compare.py \
  --model-path=/models/MiniMax-H3 \
  --tasks=ref2va --reference-video=/inputs/reference.mp4 \
  --legacy-gpus=0,1,2,3,4,5,6,7 --legacy-memory-fraction=0.4 \
  --infinilm-gpus=0,1,2,3
```

## 参考实现

- [Diffusers H3 权重转换：QKV 与 SwiGLU 顺序](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py)
- [Diffusers H3 transformer：AdaLN 与混合精度结构](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [Transformers Qwen3-VL：视觉 attention 的分段接口](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

这些链接为检查时读取的上游 main，不是本地已安装依赖的版本声明。部署时应固定版本，并保留构建日志及运行参数。
