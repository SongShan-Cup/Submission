# MiniMax-H3 推理与画质检查（2026-11-02）

## 实际运行结果

已通过用户指定 SSH 入口进入 worker-0 容器。本次入口主机为 zz-login02（入口可能负载均衡）。8 张 BW DCU，每卡约 64 GiB；PyTorch 2.9.0，HIP 6.3.26093。

实际使用代码 `/workspace/minimax-h3-upload-20261101/InfiniLM`，环境 `/workspace/h3-deps-20261101/activate-h3.sh`。Python 实际导入的 InfiniLM/InfiniCore 均来自该上传目录，运行库来自 `/workspace/h3-deps-20261101/native/lib`。没有修改或重新编译远程项目代码。

执行：

```bash
source /workspace/h3-deps-20261101/activate-h3.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /workspace/minimax-h3-upload-20261101/InfiniLM
timeout 900 python -u examples/minimax_h3.py fl2va \
  --model-path /public/home/aclsylqidf/shared/MiniMax-H3 \
  --text 'A dog falls asleep' \
  --first-frame /public/home/aclsylqidf/Golde33443.jpg \
  --gpus 0,1,2,3,4,5,6,7 --num-frames 16 --steps 2 \
  --height 256 --width 256 \
  --output /public/home/aclsylqidf/minimax-h3-audit-20261102/smoke.mp4
```

该参数仅用于定位运行故障，不用于评估画质。模型加载完成后，在图像条件编码阶段 SIGSEGV，未生成新视频。崩溃栈为：

```text
0x0000000000000000
infinicore::op::MultiheadAttention::MultiheadAttention
infinicore::op::mha
Qwen35VisionAttention::forward
Qwen35VisionBlock::forward
Qwen35VisionModel::forward_with_deepstack
Qwen3VLModel::forward
```

## 当前确认的算子问题

| 项目 | 验证结果 |
|---|---|
| MHA | 模型运行和独立小张量调用均 SIGSEGV；已确认阻塞 |
| MHA varlen / MHA KV-cache | 海光实现同样受 ENABLE_FLASH_ATTN 编译开关控制；当前构建也无法提供这两类后端。未单独执行，不能当作模型已到达的下一处报错 |
| timestep_embedding | DCU 小张量执行及同步通过 |
| index_copy | DCU 小张量执行及同步通过 |
| 其余算子 | 尚未逐项完成数值对照；不能断言全部可用或全部缺失 |

构建配置 `.xmake/linux/x86_64/xmake.conf` 和 `configure-core.log` 均确认：`hygon-dcu=true`、`aten=true`、`flash-attn=false`。

`InfiniCore/src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc` 用 `#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)` 包围实现和注册。MHA 调用时跳到空地址，与该注册缺失吻合。因此，首要问题是当前构建没有可调用的注意力后端，不是模型权重文件缺失。

特别注意：当前海光 MHA 文件虽然名字含 flashattn，run 内实际使用 matmul → softmax → matmul，展开完整注意力矩阵。只打开编译开关不足以解决高清长序列的显存问题，需接入并验证真正节省显存的注意力实现。源码目录里没有单独 hygon 子目录，也不能作为其他算子不支持海光的证据；部分算子复用 nvidia/cuda 实现。

## 已有视频为何不清晰

用户尚未指明具体视频，以下结论针对服务器上实际找到的历史生成物，不包含模型仓库 assets 中的宣传/示例视频。

| 文件 | 实际宽×高 | 帧数 | 历史日志步数 | 后端 |
|---|---:|---:|---:|---|
| minimax_h3_flash_test/fl2va_flash_2step.mp4 | 576×384 | 124 | 2 | legacy PyTorch |
| minimax_h3_flash_test/ref2va_flash_2step.mp4 | 576×384 | 124 | 2 | legacy PyTorch |
| minimax_h3_reference_test/fl2va_legacy.mp4 | 576×384 | 124 | 10 | legacy PyTorch |
| minimax_h3_reference_test/fl2va_legacy_first22.mp4 | 576×384 | 22 | 裁剪文件，未独立认定 | 未独立认定 |
| minimax_h3_reference_test/fl2va_legacy_first22_192x128.mp4 | 192×128 | 22 | 缩小文件，未独立认定 | 未独立认定 |

上述视频均为 24 fps、H.264。目录位于 `/public/home/aclsylqidf/`。分辨率、帧数和编码通过 PyAV 读取，步数和后端通过相应日志核验。

1. **生成/处理分辨率低**：576×384，另有 192×128 缩小版本，放大播放时无法恢复细节。
2. **去噪步数少**：已完成视频使用 2 或 10 步，项目说明默认 50 步。低步数是画质不足的重要嫌疑，尚未完成同种子、同分辨率的 50 步对照，不量化其贡献。
3. **低分辨率背后存在明确显存瓶颈**：历史 full_resolution_oom 在 PyTorch scaled_dot_product_attention 处申请 158.57 GiB；Ref2VA full_length_oom 申请 442.14 GiB。报错指向注意力实现的内存需求，不能通过简单增加采样步数解决。
4. **VAE 精度风险**：当前脚本在加载后对视频/音频 VAE 调用 `.to(device=execution_device, dtype=DTYPE)`，DTYPE 为 BF16。实际运行警告 encoder、decoder、quant_conv 等应保留 FP32，直接转换可能产生不一致结果。尚未做 FP32/BF16 解码对照，不能宣称它已被证明是模糊的原因。

已找到的成功视频并非 InfiniLM 原生推理成功的证据，也不能根据文件夹名 flash_test 推断它使用了原生 FlashAttention。

## 建议处理顺序

1. 修复并验证海光 MHA 的构建注册，先让独立调用和图像编码器通过，同时给未注册后端提供明确异常，避免空函数指针崩溃。
2. 为原生和 PyTorch 参照流程接入有效的节省显存注意力后端，检查长序列、BF16、非因果注意力及实际 head_dim 的支持情况，做数值对照。
3. 按依赖要求保留 VAE 必要模块的 FP32 精度，并对照解码结果。
4. 固定输入、种子、分辨率，完成 50 步短视频，再提高到目标分辨率及 124 帧，最后用 comparison 脚本比较原生/参照结果。

## 按说明文档核对正式生成流程

已阅读项目《使用InfiniLM运行MiniMax H3推理说明.md》、模型 README.zh-CN.md，以及本地 reproducible-768p-fl2va-request.sh。并核对容器实际安装的 diffusers 尺寸解析源码。

- 本项目 InfiniLM 示例为 124 帧、50 步；comparison 脚本固定 seed=42、steps=50、num_frames=124。原生生成不加 `--legacy`；该选项仅用于 PyTorch 参照。
- 不传 `--height/--width` 时，当前 diffusers 配置以短边 768、像素预算 768×1344 解析尺寸，并按 32 对齐；有首帧时继承首帧比例。已核验 Golde33443.jpg 为 800×965，按该规则应生成宽 768、高 928。无首帧时默认约 1344×768。
- 模型文档将完整系统分为 Context-IR、H3-Base、Regenerate-2K。本地基础模型对应 768p；当前仓库说明中的完整 2K 路径依赖额外托管 API，不能把离线基础推理或简单提高宽高当作同一流程。
- 官方演示用例还使用详细的结构化场景、声音和音乐提示词。项目比赛脚本则明确使用固定短提示词；比赛对比应保持其输入，不应擅自改写成官方演示提示词。
- 文档的构建示例包含 `--flash-attn`，当前海光构建实际关闭了它；应先修复后端再执行以下正式参数，避免再次触发已确认的崩溃。

正式 FL2VA 命令（待 MHA 后端修复后运行）：

```bash
source /workspace/h3-deps-20261101/activate-h3.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /workspace/minimax-h3-upload-20261101/InfiniLM
mkdir -p /public/home/aclsylqidf/minimax-h3-results
python examples/minimax_h3.py fl2va \
  --model-path /public/home/aclsylqidf/shared/MiniMax-H3/diffusers \
  --text 'A dog falls asleep' \
  --first-frame /public/home/aclsylqidf/Golde33443.jpg \
  --gpus 0,1,2,3,4,5,6,7 \
  --num-frames 124 --steps 50 --seed 42 \
  --output /public/home/aclsylqidf/minimax-h3-results/fl2va_50step.mp4
```

这里使用现有 8 张 DCU；文档示例为 4 卡。命令保留默认生成尺寸，不作低分辨率覆盖。该命令尚未成功执行，不能视为已验证的生成结果。

## 留存证据

远程持久目录：`/public/home/aclsylqidf/minimax-h3-audit-20261102/`。

本地同目录下的 `smoke.log`、`crash-backtrace.txt`、`operator-probes.json` 已取回。远程生成的约 41 GiB core.30136 位于上传代码目录；本次保留，未下载。推理进程已结束，没有留下活跃的推理任务。
