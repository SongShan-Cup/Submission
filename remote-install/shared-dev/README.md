# 共享源码长期开发环境

配置日期：2026-11-02。仅配置指定任务的 worker-0。

## 日常使用

在 SSH 登录节点修改并保存 `/public/home/aclsylqidf/Projects/minimax-h3-dev` 中的源码。
进入 worker-0 的新交互式 Bash 时，`/root/.bashrc` 自动加载本目录的 `activate-h3-dev.sh`。
已打开的 Bash 终端或者非交互式脚本，请先执行：

```bash
source /public/home/aclsylqidf/Projects/minimax-h3-dev/remote-install/shared-dev/activate-h3-dev.sh
cd /public/home/aclsylqidf/Projects/minimax-h3-dev
```

`infinicore` 和 `infinilm` 的现有 editable 安装已指向共享项目中的对应 Python 包；无需每次手动设置 PYTHONPATH。
脚本同样应从共享目录运行。不要继续运行 `/workspace/minimax-h3-upload-20261101` 中的旧脚本。
Python 源码修改后重新启动程序。已经运行的进程不会自动替换已导入的模块。

## 编译代码与环境重建

本次继续复用已有原生库，没有重新编译。C++、HIP、算子以及绑定接口修改后，应在容器内重新配置、编译并安装相应库。
源码目录中的复制构建缓存可能包含旧绝对路径，重新构建时需检查构建配置与最终加载的 .so 路径。

依赖环境、原生库和编译工具仍位于 `/workspace/h3-deps-20261101`。共享源码长期保存不等于容器环境已持久化。
容器重建时，需要先恢复依赖环境，再配置自动激活；不能只依靠源码目录恢复整个环境。
共享目录的配置脚本及备份可用于重新配置。`bind-shared-sources.py` 仅迁移当前版本的已有 editable 安装，不负责首次安装依赖。
重装这两个 Python 包后，应再次核对实际导入路径。

## 变更与验证

- 迁移 InfiniLM 的 editable .pth 路径。
- 迁移 InfiniCore 的 editable finder 路径，包括测试包路径。
- 同步两个包的 direct_url.json 和相关 RECORD 校验信息。
- 配置 worker-0 的 /root/.bashrc 自动激活；不自动改变当前目录。
- 激活脚本设置 PYTHONDONTWRITEBYTECODE=1，避免容器 root 向用户拥有的 NFS 源码目录写入字节码缓存。
- 已在新交互 Bash 中通过 Python 隔离模式实际导入并验证两个包的路径与安装元数据；InfiniCore 原生扩展成功加载。
- 未进行模型推理、重新编译或依赖升级。

原配置备份：本目录下 `backup-20261102-9GxEm5/container-config.tar`，内容包括修改前的 .bashrc、editable 配置、direct_url.json 和 RECORD。
该备份不包含完整依赖环境。
