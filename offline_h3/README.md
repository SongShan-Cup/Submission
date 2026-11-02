# H3 离线依赖上传可行性检查

> **2026-10-30 更新：平台已向用户确认依赖需要自行配置。当前入口为 [下载与安装说明](下载与安装说明.md)，含本地下载脚本、包清单及个人虚拟环境安装方法。**
> 下面以及此前流程复核中的“先等待管理员发布环境”不再作为当前操作要求。

> 本文下面保留的是早期排查记录，已被新增文档和环境信息修正，不能继续作为当前安装流程。
> 当前已有 4 GiB 个人 AFS，512 MB 不能再视为所有存储的合计上限；InfiniFabric 的依赖供应与原始 Pod 不同。
> 请以 [平台操作流程复核](平台操作流程复核.md) 为准。此前未成功下载任何依赖包。

检查日期：2026-10-29。

## 当前结论

用户已确认：512 MB 是总存储额度，模型权重由平台另行挂载，不计入此处的上传方案。

因此不能只按压缩包体积设计方案。必须将项目已有文件、压缩包、解压文件、安装结果、缓存、构建中间产物和测试输出同时纳入实际计费存储范围。当前不建议在该额度内执行完整源码构建；优先使用镜像预装、外部挂载的依赖，或在匹配环境中预编译后仅部署必要运行文件。能否将最小运行文件装入剩余额度，仍需实际测量，不能仅由 512 MB 数字得出绝对结论。

当前目录没有下载好的安装包。实际访问 PyPI 和 GitHub 均失败，错误为 `Temporary failure in name resolution`。没有将未验证的包或其他平台的二进制伪装为可用依赖。

## 已知目标环境

- Python 3.11.9。
- PyTorch 发行版：`2.9.0+das.opt1.dtk2604`；模块报告 `2.9.0`。
- DTK 路径：`/opt/dtk -> /opt/dtk-26.04`。
- HIP 版本：`6.3.26093`。
- `hipcc` 可以执行。
- PyTorch 能识别 8 张名称为 `BW` 的设备。
- 当前解释器未找到 InfiniCore、InfiniLM、Diffusers、Accelerate、ninetoothed、ntops；PATH 中未找到 xmake。
- 已有 Transformers 4.57.6、safetensors 0.7.0、setuptools 80.8.0。
- Linux 发行版、glibc、CPU 架构、PyTorch C++ ABI、实际 GPU 架构，以及临时目录和其他挂载是否独立于总存储额度，尚未确认。

## 权重容量：不能纳入 512 MB 上传包

根据当前模型索引中的 `metadata.total_size`：

| 组件 | 声明的权重字节数 |
| --- | ---: |
| 根目录 transformer | 66,280,430,080 |
| text_encoder | 66,714,780,128 |
| 视频 vae | 10,415,475,936 |
| 三项合计 | 143,410,686,144 |

合计约 143.4 GB（133.6 GiB），尚未计入音频 VAE。这里是索引声明的张量总量，不是本地已有权重文件大小。当前工作区没有相应的索引分片。用户已确认平台将另行挂载权重，因此这部分不占计划中的 512 MB 项目存储。

## 应获取的资源及可行性条件

| 资源 | 获取/准备方式 | 验证要求 |
| --- | --- | --- |
| H3 版 Diffusers | 官方验证版本或固定 commit 的源码/wheel | 必须包含 H3 transformer、modular pipeline、引用素材接口及所需 VAE/scheduler；不能仅依据 `>=0.32.2` |
| Accelerate 和传递依赖 | 对应目标环境的离线 wheels | 依赖解析须保留镜像已有 DTK PyTorch，不能引入通用 torch/NVIDIA 包 |
| 媒体和比较组件 | AV、Pillow、NumPy、scikit-image 等及必要传递依赖 | Python、平台和 glibc 标签匹配；包含所需编码支持 |
| InfiniLM 间接导入依赖 | tqdm、janus、xxhash 等 | 以完整导入检查核对；不只检查示例文件的直接 imports |
| xmake | 平台匹配的发行包或平台预装版本 | 可在当前权限和 CPU 架构下执行 |
| xmake 构建依赖 | 与其包配方匹配的 Boost、pybind11 等缓存/源码资源 | Python 的 pybind11 wheel 不等同于 xmake 依赖已经满足；离线不得继续请求下载 |
| 海光 FlashAttention | 平台提供、或在匹配 DTK/PyTorch 环境构建 | 该基线链接 `flash_attn_2_cuda*.so` 和 `flash_attn/lib/libflash_attention.so`；普通 CUDA wheel 不能替代 |
| 通信与 DTK 兼容库 | 优先使用镜像已有 SDK 与通信库 | 现有海光构建链接 `nccl` 等兼容接口，需确认头文件/库的位置 |
| InfiniCore / InfiniLM | 使用已有指定基线源码构建 | 原生扩展、底层库、ABI 和设备架构一致 |
| 九齿、ntops、后端 | 后续单独准备匹配平台的开发依赖 | 初次运行现有非九齿路径可暂不纳入，最终开发阶段仍需补齐 |

当前海光构建还调用 DTK CUDA 兼容层的 `nvcc` 和 `clang/clang++`，不能由 hipcc 可用推断整个构建链已经完整。FlashAttention 默认路径有 Python 3.10 的历史值，当前 Python 3.11 环境需指定真实路径。

## 控制上传体积的方法

1. 不打包模型权重、PyTorch、完整 DTK SDK、已有项目源码和 NVIDIA 依赖。
2. 第一批仅准备当前推理链需要的软件；九齿开发依赖后续单独准备。
3. 优先获取官方匹配的 InfiniCore/FlashAttention 构建产物，或使用平台安装的工具和构建缓存，避免上传整套构建依赖。
4. 按目标环境的包清单解析依赖；不要假设本地电脑的 Python 或系统平台等同于服务器。
5. 所有文件获取后，以实际字节数核算整个计费目录，采用保守上限 512,000,000 字节，并为输出和日志保留余量。
6. 同时核算解压、安装和构建的临时空间。压缩包小于上传限制，并不保证在相同总存储额度中可以安装。
7. 交付包应包含固定版本清单、来源、SHA256、安装顺序及未解决事项。只有完成相应检查后才标记为可运行。

## 本轮下载与缓存检查结果

访问以下资源均因 DNS 解析失败，未取得响应：

- https://pypi.org/pypi/diffusers/json
- https://api.github.com/repos/huggingface/diffusers/branches/minimax-h3
- https://api.github.com/repos/xmake-io/xmake/releases/latest
- https://api.github.com/repos/SongShan-Cup/infiniCore/contents

本地 pip 缓存没有 Diffusers、Accelerate、xmake 或所需海光 FlashAttention。存在部分普通 Python wheels，但不足以组成完整依赖闭包，部分还是 CPython 3.9 或不匹配的 GPU 二进制，因此未复制进上传目录。

## 下载环境与用户网络的区别

此次 DNS 失败发生在助手运行命令的受限执行环境中。当前会话明确限制网络访问；失败并不能证明用户电脑、聊天连接或比赛平台整体断网，也不能仅凭错误区分沙箱限制和该执行环境的 DNS 配置问题。没有进行解除隔离、修改系统 DNS 或越权联网操作。

## 下一步的必要条件

- 确认 512 MB 配额之外是否有可写的临时构建空间或可挂载的依赖目录；权重另行挂载已确认。
- 取得官方支持 H3 的 Diffusers 版本和匹配的海光 FlashAttention、通信/构建配置。
- 恢复本地下载环境对 PyPI/GitHub 的访问，或提供可直接读取的官方离线包。

满足以上条件后，再下载、验证和统计实际安装包体积；目前不能给出可信的“完整包小于 512 MB”承诺。
