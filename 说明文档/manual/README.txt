InfiniFabric 管理员与用户操作手册
版本 0.10.0-dev.42，文档日期 2026-10-29
文档修订 release；对应 0.10.0-dev.42，部署状态见手册首页。
打开 HTML 阅读，可用浏览器打印为 PDF。HTML 内含样式，不依赖网络。
先读第 0 节：管理员从第 3 节开始；推荐用户先按第 31 节编译并提交整节点批作业。
独占 Bash 见第 8 节，交互开发见第 9 节；试用检查表见第 28 节和 trial-checklist.csv。
examples 文件可多选上传到平台，上传 ZIP 不会自动解压。NumPy 示例在单独目录内运行。
CMake 路线使用 examples/batch_demo 的四个文件，注意其 run.sh 与 examples/run.sh 不同。
trial_control.py 用于短时排队、失败、取消及到期；运行仍独占计算节点全部卡。
DCU 示例来自项目既有示例，请在目标节点检查实际运行结果。
