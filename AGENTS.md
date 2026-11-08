# MiniMax-H3 项目长期约定

## 执行环境

- 后续项目的开发、环境检查、依赖安装、编译、测试和模型推理均在下述远程服务器及容器中进行。
- 本地用于下载所需文件、传输文件，以及维护项目说明和长期指令。
- 远程服务器无法联网。所需代码、安装包和其他资源必须先在本地下载，再传输到服务器；不要依赖服务器直接联网下载。

## 登录方式

先从本地 Windows 通过 SSH 登录服务器：

```powershell
ssh -i "C:\Users\xuyh0\.ssh\aclsylqidf_zzeshell" -p 65032 aclsylqidf@zzeshell.scnet.cn
```

再在服务器上进入项目容器：

```bash
kubectl exec -it vcjob-2116206198177124354-aclsylqidf-33085-worker-0 -n aclsylqidf /bin/bash
```

以上为用户提供的交互式登录命令；自动执行时可根据终端情况调整交互参数，保持服务器、命名空间和容器目标一致。

## 模型参数位置

模型参数位置由 `remote-install/h3-paths.env` 的 `H3_MODEL_ROOT` 配置。
迁移服务器或更换目录时，先更新该配置文件，再运行或构建。

## 远程工作目录与 Git 约定（2026-11-02 用户更新）

- 后续统一在 `remote-install/h3-paths.env` 的 `H3_PROJECT_ROOT` 指向的项目目录下开展项目工作；代码修改、检查报告、运行日志、测试结果及生成视频均保存到此目录的相应子目录，避免继续散落在 `/workspace` 或用户目录的其他位置。
- 该目录由用户使用 Git 管理。修改前先检查仓库状态，并创建新的工作分支；保留用户已有修改，不覆盖或重置用户工作。
- 本地可维护长期说明及必要副本，正式结果应直接写入上述远程项目目录。
- 模型权重仍使用上文指定的共享目录；已安装的依赖环境可继续复用，但必须核验实际加载的代码及运行库与当前工作目录一致。

## 共享源码开发环境（2026-11-02 已配置）

- worker-0 中现有 `InfiniCore`、`InfiniLM` 的 editable 安装已改为加载 `H3_PROJECT_ROOT` 指向的源码；新开的容器交互式 Bash 会自动激活 H3 环境。
- 已打开的容器终端及非交互式脚本应先执行 `source "$H3_PROJECT_ROOT/remote-install/shared-dev/activate-h3-dev.sh"`，再从共享项目目录运行脚本。
- Python 源码保存后重启程序生效；C++、HIP、算子修改后仍需在容器内重新配置、编译并安装，核对最终加载的库。依赖环境位置由 `H3_DEPS_ROOT` 配置；容器重建或依赖目录迁移后需要恢复环境并运行 `remote-install/shared-dev/repair-relocated-env.py`。
- 配置、原配置备份及说明位于远程项目 `remote-install/shared-dev/`；验证记录位于远程项目 `audit/共享源码长期开发配置-20261102.md`。首次配置验证了两个包实际从共享目录导入及 InfiniCore 扩展加载，未执行模型推理。

## 维护

- 用户更新服务器、容器、联网条件或模型路径时，同步更新本文件。
- 本文件记录连接方式和私钥路径，不存放私钥内容。

## 已核验的存储与任务信息（2026-11-02）

- SSH 登录入口为 `zz-login01`；worker-0 实际运行在计算节点 `10.5.15.78`。同任务还存在 worker-1，运行在 `10.5.16.27`。
- 容器中的 `/public/home/aclsylqidf` 与 `/root/private_data` 是同一份 NFS 用户存储的两个挂载入口，登录节点也能访问该用户目录。需要长期保留的代码、日志、输出优先放在此用户目录下。
- `/workspace` 属于容器 overlay 文件系统，不能将其当作独立持久存储。目前 `/workspace/minimax-h3-upload-20261101` 包含代码，`/workspace/h3-deps-20261101` 包含依赖及构建产物。重建前需确认已备份或保存镜像。
- 共享存储还存在 `Projects/InfiniCore`、`Projects/InfiniLM` 和 `minimax-h3-upload-20261101` 上传目录；尚未核验这些目录与容器内版本是否一致。
- 容器由 Volcano Job `vcjob-2116206198177124354-aclsylqidf-33085` 管理，worker-0 为 120 CPU、472Gi 内存、8 张 DCU。启动脚本最终执行 `sleep infinity`，实际 PID 1 已核验；`kubectl exec` 只是进入已经运行的容器。
- 详细检查记录见 `audit/服务器与容器存储检查-20261102.md`。当前账号无权读取上层 Volcano Job 完整配置。
