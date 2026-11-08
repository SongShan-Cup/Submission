# H3 NineToothed / Hygon 验收检查点

本目录对应 `codex/h3-ninetoothed-attention-20261103` 分支，基线提交 `afe1e88`。
本轮只接入并验收 MHA，不代表整个比赛适配已经完成。

## 接入路径

InfiniLM → InfiniCore MultiheadAttention → `mha_ninetoothed_hygon.cc` →
`libh3_ninetoothed.so` → NineToothed arrangement/application 生成代码 → Triton HIP 编译链 → gfx936 HSACO。
计算定义位于 `InfiniCore/src/infinicore/ops/multi_head_attention/ninetoothed/attention.py`。
`build_hygon.py` 只补充 AOT 编译和主机启动适配，没有手写 Triton/CUDA 设备计算。

- 单独的 xmake 开关 `--hygon-ninetoothed-mha=y` 选择此 MHA 路径；旧的通用 `--ninetoothed` CUDA 算子包保持关闭。
- 开关启用时不注册原 Hygon 厂商 MHA。未知形状、数据类型或 ALiBi 明确报错，不静默回退。
- 支持 BF16/FP16、head_dim 72/96/128、batch、GQA、不同 Q/K 长度、lower-right 因果遮罩。
- 分块在线 softmax，不分配 S×S 中间张量。支持 Q=K=V、正向非连续步长；最内层非连续输入/输出通过既有 ATen 张量复制适配。
- 每个线程按设备编号＋上下文分别加载内核，沿用 InfiniCore 的流。导出的调用计数用于核验真实执行路径。
- 固定适配当前 NineToothed 0.26.0、海光 Triton 3.5.1+das.opt1.dtk2604.torch290、DTK 26.04。
  NineToothed 公共 AOT 仍假定 CUDA，因此适配器使用其 CodeGenerator 及 Triton AST 编译接口。
  DTK 后端不识别新版 LLVM GEP 的 nuw/nusw 标记，仅在上述海光版本移除这些优化承诺；计算语义不变。

## 构建和选择环境

在远程容器中先按目标机器修改 `remote-install/h3-paths.env`。若项目或依赖目录
是从旧服务器移动过来的，先运行：

```bash
"$H3_DEPS_ROOT/venv/bin/python" remote-install/shared-dev/repair-relocated-env.py
```

然后在项目根目录运行：

```bash
bash remote-install/h3-ninetoothed/build.sh
source remote-install/h3-ninetoothed/activate.sh
```

独立安装目录为 `remote-install/h3-ninetoothed/native`。原 `h3-fix/native` 的已验收库不被覆盖。
项目默认激活脚本仍指向原基线；运行新路径必须 source 本目录 activate.sh。
构建脚本保留项目既有厂商库依赖，其他算子仍可使用原有基础实现；这不是“全模型去厂商依赖”的声明。

## 本轮验收

```bash
source remote-install/h3-ninetoothed/activate.sh
python remote-install/h3-ninetoothed/check_attention.py
python remote-install/h3-ninetoothed/replay-first-step.py
bash remote-install/h3-ninetoothed/run-fl2va.sh
```

- 算子：FP32 参考，BF16 atol/rtol=0.025/0.025；FP16=0.004/0.01。
- 集成：冻结的相同首步输入，与 `afe1e88` 的正确同步版本输出比较；本轮工程门槛为相对 RMSE<5%、cosine>0.998、数值有限、八卡都实际调用九齿。
- 推理：固定狗图、文本、seed42、124 帧、50 steps、八卡，使用原示例入口，生成输出并按既有比较方法核对规格与相似度。
- 内存：分别记录 Torch 分配器峰值、设备可用显存差；完整推理每 0.5s 采样整卡使用量，不把这些混称为精确整卡峰值。
- `audit/20261103-ninetoothed` 保存日志、JSON、基线信息、验收报告。`outputs/20261103-ninetoothed` 保存视频。
- 脚本的默认输出路径对应本轮；重复验收前应另设输出或归档，避免覆盖本轮证据。

本轮结束后停在验收检查点。此前图像条件编码误差、其他新增算子合规映射、正式评测输入及阈值、性能优化仍需后续处理。

若要重新构建旧厂商 MHA，必须先在 InfiniCore 中配置 `xmake f --hygon-ninetoothed-mha=n`，再使用旧构建脚本。仅切换运行环境不需要重新构建。
