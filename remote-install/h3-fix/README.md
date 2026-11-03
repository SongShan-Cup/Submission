# MiniMax-H3 海光推理修复与复现

工作分支：`codex/h3-attention-quality-fix-20261102`。所有命令均在 worker-0 容器内执行，服务器无需联网。

## 已验证运行

```bash
cd /public/home/aclsylqidf/Projects/minimax-h3-dev
source remote-install/shared-dev/activate-h3-dev.sh
unset MINIMAX_H3_TRACE_STAGE
bash remote-install/h3-fix/run-fl2va.sh
```

默认使用共享模型权重、`/public/home/aclsylqidf/Golde33443.jpg`、提示词 `A dog falls asleep`、seed 42、8 卡、124 帧、50 步。保持输入比例后输出 768×928、24 fps。结果为 `outputs/20261102-fix/fl2va_50step.mp4`。覆盖参数可追加在脚本后，例如 `--output /绝对路径/文件.mp4`。注意默认路径已有本次验收视频，重新运行可指定新名称。

PyTorch 参照：追加 `--legacy --output /绝对路径/参照视频.mp4`。

## 重编译

```bash
bash remote-install/h3-fix/build.sh
bash remote-install/h3-fix/build-lm.sh
```

InfiniCore 安装到项目内 `remote-install/h3-fix/native`，InfiniLM 扩展安装到 `InfiniLM/python/infinilm/lib`。共享激活脚本优先使用该版本；`native/` 不纳入 Git。Python 依赖仍复用容器 `/workspace/h3-deps-20261101`，容器重建后需恢复该环境。

## 数值检查

```bash
source remote-install/h3-fix/activate.sh
python remote-install/h3-fix/check_attention.py
python remote-install/h3-fix/check-h3-ops.py
python remote-install/h3-fix/inspect-video.py
```

`capture-first-step.py` 在保持原函数签名的前提下记录真实输入和输出；`replay-first-step.py` 用同一输入调用 PyTorch 参照。相关 `.pt` 中间文件不纳入 Git。

`trace-reference.py` 与 `trace-native.py` 用于逐层排查，必须先重新运行参照采集，保证两端使用同一 `native-first-input.pt`。C++ 的 `MINIMAX_H3_TRACE_STAGE=embeddings` 或 `block:N` 仅用于返回调试隐藏状态；**正式视频推理必须取消该变量**。

带具体历史 PID 的 `finish-validation.sh`、`run-sync-validation.sh` 等为本次执行记录，不是后续通用入口。后续推理使用 `run-fl2va.sh`。

完整结论见 `audit/20261102-fix/修复与验收报告-20261103.md`。
