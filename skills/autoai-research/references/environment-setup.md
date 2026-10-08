# AutoAI 环境配置与修复手册

用于首次使用、切换机器/解释器，以及依赖缺失、导入失败或计算服务异常后的恢复。先检查，再通知用户发现的问题和准备采取的修复，随后执行并复查。环境准备不能跳过数据与实验设置的第一轮汇报、选择；环境检查不启动训练。

## 1. 实际需要安装什么

依赖以当前代码随附的文件为准，不凭模型名称猜测，也不要求用户安装整个开发环境。

| 使用场景 | 依赖来源 | 作用 |
|---|---|---|
| 技能客户端、连接已有计算服务 | 技能根目录 `requirements.txt` | `httpx>=0.27,<1` 负责服务请求；`openpyxl>=3.1,<4` 负责 Excel 读写 |
| 本地计算 | 项目 `backend/requirements.txt`，加技能 `requirements.txt` | FastAPI/Uvicorn、上传解析、NumPy/Pandas/SciPy、绘图、scikit-learn、XGBoost、PyTorch、rampy 等 |
| 需要复用项目已有版本基线 | 项目 `backend/constraints-verified.txt` | 约束文件记录曾验收的直接依赖版本，不是额外的安装清单，也不代表每台机器已通过检查 |
| 开发测试 | 项目 `backend/requirements-dev.txt` | 包含 pytest 等；普通技能使用无需安装 |

Python 3.12、64 位是项目的验收基线。CSV 检查可使用标准库；Excel 需要 openpyxl。当前本地后端在传统模型训练路径中也导入 PyTorch，所以只安装 scikit-learn 不足以运行本地 SVM/PLS-DA。

GPU 不是首次使用的必需条件。已有可用 CUDA PyTorch 时保留它；新环境只跑快速传统模型时可用 CPU 版本。技能公开九模型不需要 AggMap、mamba-ssm、Docker、Node.js 或前端构建工具。Git 用于保存代码版本；没有 Git 或使用源码压缩包时允许继续，代码提交信息会为空。

## 2. 首次使用：先检查并通知

1. 确定使用已有远程计算服务还是本地计算。读取技能根目录 `runtime.json` 中的 `python`、`project_dir`；用户指定路径优先。配置只能保存本机路径，不能包含凭据。旧机器的路径不存在时重新定位，不沿用示例中的用户目录。
2. 本地计算需要完整 AutoAI 项目源码，至少有 `backend/requirements.txt`、`backend/app/main.py`、训练预检和 Worker 模块及 `run_classic.py`。从当前工作区、输入文件附近或已配置路径定位。缺少源码且来源不明时集中询问项目位置/来源，不把只有数据的文件夹当项目，也不安装同名未知软件包代替源码。
3. 找到可执行的 Python 后，先运行下面的标准库检查器。它在目标解释器中核对依赖版本、相关传递依赖、关键模块导入和 PyTorch CPU/CUDA 状态，不安装软件，不启动服务，不读取凭据或上传数据。

```text
<可用Python> <技能目录>/scripts/check_environment.py --python <目标Python> --mode local --project-dir <AutoAI项目> --output <任务目录>/environment-before.json
```

只连接已有服务时用 `--mode client`，不要求本机有后端或 GPU。返回 `ready=true`、退出码 0 表示本次依赖和导入检查通过；退出码 1 表示存在阻断项。它不代替后续的服务检查。非 3.12 Python 标为尚未按项目基线验证，若安装或导入失败，优先使用 3.12 专用环境。

4. 发现问题先用一句清楚的说明通知用户，然后按后文修复。例如：“当前环境缺少 Excel 读取依赖，我会补齐后继续检查数据。”“连接计算服务的依赖版本冲突，我会修复兼容版本并复查。”“现有环境有多项相互冲突的依赖，我会为本技能建立专用环境。”不把异常栈或安装命令当成用户待办，不静默宣称环境已经就绪。
5. 复用健康环境。缺失少量依赖时只补所需项；共享环境存在多项冲突时优先创建专用环境，避免整体升级/降级共享环境。只有真正缺少项目来源、写入权限、远程凭据或管理员权限等无法自行解决的信息才请求用户协助。

## 3. Windows：建立专用本地环境

以下 PowerShell 示例中的项目目录须替换为真实路径。`work/autoai-env` 位于项目已忽略的工作目录，不能提交到 Git。若该环境已存在，先检查并复用；不用 `--clear` 清空它。没有 Python 时先定位已有 Conda/系统/工具运行时，必要时从 [Python 官方 Windows 文档](https://docs.python.org/3.12/using/windows.html)选择用户级安装。

```powershell
$projectDir = 'D:\PythonProject\AutoAI'
$skillDir = Join-Path $env:USERPROFILE '.codex\skills\autoai-research'
Set-Location -LiteralPath $projectDir
py -3.12 -m venv 'work\autoai-env'
$autoaiPython = Join-Path $projectDir 'work\autoai-env\Scripts\python.exe'
& $autoaiPython -m pip install --upgrade pip
```

若没有 `py` 启动器，改用已找到的 Python 3.12 完整路径运行 `-m venv`。所有安装、检查和启动均使用 `$autoaiPython`，避免裸 `pip` 安装到另一个解释器。直接调用环境内的 Python 即可，无需修改 PowerShell 执行策略或激活脚本；依据 [Python venv 官方说明](https://docs.python.org/3.12/library/venv.html)。

新建 CPU 环境时先安装项目已有基线的 PyTorch，再安装其余依赖：

```powershell
& $autoaiPython -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
& $autoaiPython -m pip install -r (Join-Path $projectDir 'backend\requirements.txt') -r (Join-Path $skillDir 'requirements.txt') -c (Join-Path $projectDir 'backend\constraints-verified.txt')
```

CPU 下载来源及版本命令已对照 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)。本项目不使用 torchvision/torchaudio，无需一并安装。若基线约束在目标平台没有可用 wheel，核对平台后可使用 requirements 的兼容版本范围，并记录实际版本与验证结果；不能悄悄降低项目依赖范围。

需要 NVIDIA GPU 时先检查驱动和当前 PyTorch 的 `cuda_available`。缺少可用 CUDA 版本时根据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)或同版本历史页面选择与驱动兼容的 wheel，再安装上面的两份 requirements。不要把已经可用的 GPU 环境换成 CPU 环境，也不凭 `nvidia-smi` 的 CUDA 显示值直接猜 wheel。

## 4. Linux/macOS 与仅客户端环境

Linux/macOS 的专用环境用 `python3.12 -m venv work/autoai-env`，解释器是 `work/autoai-env/bin/python`。同样用该解释器运行 `-m pip`。Linux CPU 安装可用上面的 CPU 索引；macOS PyTorch 从官方对应平台安装渠道获取，不使用 Linux/Windows 的 CPU 索引命令。

已有计算服务时，只需在选定解释器中安装技能依赖：

```text
<目标Python> -m pip install -r <技能目录>/requirements.txt
<可用Python> <技能目录>/scripts/check_environment.py --python <目标Python> --mode client --output <任务目录>/environment-after.json
```

此时不要为了客户端补装本地深度学习依赖。服务端地址通过 `AUTOAI_SERVICE_URL` 提供；凭据仅通过已有安全环境变量 `AUTOAI_SERVICE_TOKEN` 提供，不写入 runtime.json、命令行、URL 或日志。远程连接失败不能擅自改成另一队列的本地训练。

## 5. 按诊断修复

| 发现的问题 | 修复与复查 |
|---|---|
| Python 路径过期，或命令指向另一环境 | 核对 `sys.executable`；定位可用 3.12 解释器；更新本机路径并对同一解释器重检 |
| 缺少 pip | 对所选解释器运行 `-m ensurepip --upgrade`，然后重检；若该 Python 发行版不提供 ensurepip，按其官方方式配置或建立支持 pip 的专用环境 |
| 缺少客户端依赖 | 在所选解释器安装技能 `requirements.txt`，不要求整个开发依赖集合 |
| 缺少本地运行依赖 | 在专用环境安装两份 requirements；先决定并安装合适的 PyTorch |
| 单个相关依赖版本冲突 | 核对上游允许范围，先用 `pip install --dry-run --report <本地报告>`检查最小修改；通知用户后只修复冲突项，再检查完整相关依赖链 |
| NumPy/SciPy/PyTorch 导入或 DLL 错误 | 核对 64 位、解释器及 wheel 平台；按 requirements/约束重建专用环境并复查导入，不把 ABI/DLL 问题当作数据错误 |
| 安装超时、代理或 TLS 错误 | 保留日志，检查当前网络及代理是否可用，使用可信官方或用户已授权镜像有限重试；不关闭证书验证，不打印带凭据的代理/索引配置；相同失败重复两次后说明阻断原因 |
| `pip check` 报告其他项目冲突 | 保存报告，区分技能实际依赖链和无关包；不为无关项目补装或降级。公开九模型不需要 AggMap；其他任务确需它时按项目 `deploy/server_deploy.md` 的专门安装步骤处理 |
| Pandas 提示旧版可选 numexpr | 属于导入警告，不能直接等同训练失败；若确需消除，在确认用途后更新专用环境中的兼容版本，不为警告改动整个共享环境 |
| 端口占用、服务/Worker 过旧或不可用 | 依照执行手册的 `ready` 流程，在空闲端口启动本任务独立服务和 Worker；不结束其他用户服务或启动既有队列 |
| 已提交任务后服务失联 | 保留原任务目录、编号和队列，恢复该任务；不新建任务重复训练 |

例如 `httpcore==1.0.2` 要求 `h11<0.15`，而已安装 `h11==0.16.0` 时，如果当前 httpx 允许 `httpcore==1.*`，可以先核对 `httpcore==1.0.9` 的安装计划，再只更新 httpcore；不能先把 h11 降级而破坏其他包。使用兼容范围和最少必需变更的原则见 [pip 官方升级策略](https://pip.pypa.io/en/stable/user_guide/#only-if-needed-recursive-upgrade)。

## 6. 修复后记录与验收

1. 使用同一解释器重跑 `check_environment.py`，保存 `environment-after.json`。与修复前报告一起保留，说明解决了哪些阻断项、还有哪些非阻断警告。
2. 本地计算再运行 `python -m pip check`，保存结果并按上述规则解释共享环境中的无关冲突；新建干净环境应通过检查。保存包版本清单、Python/平台版本、依赖文件指纹和实际安装命令。`pip freeze` 出现含凭据的安装 URL 时先脱敏，不能把原文写入交付物或显示给用户。
3. 检查通过后再将确定的绝对路径保存到技能根目录 `runtime.json`，内容仅为 `project_dir`、`python`。该文件不复制到其他机器当作通用模板，也不纳入 Git。
4. 用目标解释器调用 `autoai_client.py ... ready`，随后读取 capabilities，核对服务与执行进程可用且兼容、所选模型可用。这些步骤不上传数据或提交训练；不拿 API 存在或 pip 安装成功代替实际可用性检查。
5. 告知用户“环境已配置并通过检查”，仅在以上条件满足后这样表述。普通建模请求仍等待第一轮实验选择的回复；明确直接运行或已经回答设置的任务继续执行。

如果系统条件仍无法满足，说明具体缺失项及需要用户补充的内容，保留文件、推荐方案与日志。不要在同样的安装失败上无限重试，也不宣称已成功配置。
