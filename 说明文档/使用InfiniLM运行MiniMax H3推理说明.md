# 使用InfiniLM运行MiniMax H3推理说明

# 一、模型

Minimax\-H3

https://huggingface\.co/MiniMaxAI/MiniMax\-H3

# 二、代码仓库

InfiniCore: `minimax-h3`

InfiniLM: `minimax_h3`

## Clone代码

```Python
git clone --recursive --branch minimax-h3 \
  https://github.com/InfiniTensor/InfiniCore.git

git clone --recursive --branch minimax_h3 https://github.com/InfiniTensor/InfiniLM.git
```

## 切换分支

```JavaScript
git -C ~/Projects/InfiniCore fetch origin
git -C ~/Projects/InfiniCore checkout minimax-h3
git -C ~/Projects/InfiniCore reset --ff-only origin/minimax-h3

git -C ~/Projects/InfiniLM fetch origin
git -C ~/Projects/InfiniLM checkout minimax_h3
git -C ~/Projects/InfiniLM reset --ff-only origin/minimax_h3
```

## 编译安装

- InfiniCore

```Python
cd InfiniCore

xmake f --nv-gpu=y --ccl=y --cudnn=y \
  --cuda="$CUDA_HOME" \
  --aten=y \
  --flash-attn="third_party/flash-attention" \
  -m release -cv

xmake build
xmake install
xmake build _infinicore
xmake install _infinicore
python -m pip install -e .
```

- InfiniLM

```Python
cd InfiniLM

xmake f -m release -cv
xmake build _infinilm
xmake install _infinilm
python -m pip install -e .
```

# 三、MiniMax脚本

`InfiniLM/examples/minimax_h3.py`

## FL2VA （通过首帧/尾帧\+文字引导生成视频）

```JavaScript
cd InfiniLM

python examples/minimax_h3.py fl2va \
  --model-path=MiniMax-H3/diffusers \
  --text="A dog falls asleep" \
  --first-frame=XXX.jpg \
  --gpus=0,1,2,3 \
  --num-frames=124 \
  --steps=50 \
  --output=XXX.mp4
```

`--first-frame=` 首帧图片文件路径

`--text=` prompt文本

`--legacy` 使用纯pytorch参照实现运行



## Ref2VA （通过参照视频引导生成新视频）

```Bash
python examples/minimax_h3.py ref2va \
  --model-path=MiniMax-H3/diffusers \
  --text="Transform the reference video into a colorful hand-drawn cartoon animation style" \
  --reference-video="XXX.mp4" \
  --gpus=0,1,2,3 \
  --num-frames=124 \
  --steps=50 \
  --output=XXXXX.mp4
```

`--reference-videoe=` 视频参照文件路径

`--text=` prompt文本

`--legacy` 使用纯pytorch参照实现运行，如内存不够可增加gpu数量

# 四、评分脚本

`InfiniLM/examples/minimax_h3_compare.py`

默认会跑4个task，标准答案（torch）的fl2va和ref2va，以及InfiniLM版本的fl2va和ref2va，最后会将两组视频分别对比，输出相近程度

```Bash
python examples/minimax_h3_compare.py --first-frame='XXX.jpg' --legacy-gpus=0,1,2,3,4,5,6,7 --legacy-memory-fraction=0.4
```

`--tasks=fl2va,ref2va` 指定运行任务，如果运行fl2va或两个任务都运行则需提供首帧文件，如果只运行ref2va则需提供视频输入。默认两个都运行。

`--first-frame=` 首帧文件

`--reference-video=` 视频输入



```Bash
python examples/minimax_h3_compare.py \
  --tasks=fl2va,ref2va \
  --reference-fl2va-video=minimax_h3_comparison/reference_fl2va.mp4 \
  --infinilm-fl2va-video=minimax_h3_comparison/infinilm_fl2va.mp4 \
  --reference-ref2va-video=minimax_h3_comparison/reference_ref2va.mp4 \
  --infinilm-ref2va-video=minimax_h3_comparison/infinilm_ref2va.mp4
```

脚本支持提供之前的结果视频，这样相应的task会被跳过。

结果示例：

```Bash
[result] MiniMax-H3 video comparison

  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 93.584%

  REF2VA
    Size:         PASS (PyTorch 1344x768, InfiniLM 1344x768)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 5.523%
```

## 评分规则

1. 初赛：脚本可以正常运行通过（仅FL2VA任务即可）

2. 决赛：

    1. 脚本必须能正常运行FL2VA任务，否则0分

    2. 视频大小、长度一致，20%

    3. FL2VA视频相似度，30%

    4. REF2VA视频相似度，30%

    5. 【主观分】性能、代码质量等，20%



