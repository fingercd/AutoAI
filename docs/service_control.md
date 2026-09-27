# Linux release 控制入口

`scripts.service_control.Controller` 是正式控制实现。现场适配器仅提供 Config
和 `environment(release)` 回调；回调通过既有私密配置注入环境，不能将环境、
token 或认证文件内容输出到日志。控制器只使用标准库，不导入训练实现。

```python
from scripts.service_control import Config, Controller
controller = Controller(Config(base=deployment_base, python=application_python,
                               legacy_bindings=audited_exact_bindings), environment)
result = controller.run(action)  # status / start / stop / restart
```

base 下包含 current、storage、runtime；release 必须是干净 Git checkout，
release/storage 必须指向持久 storage。适配器应从固定的已验收 release 导入
控制模块，而不随 current 变化，因此回退到没有此模块的旧 release 仍能控制。
控制版本为 release-controller-v1；进程记录保存实际 app HEAD/tree、路径、
Web/worker 的 PID、start_ticks、UID、cwd、状态和操作时间。

操作使用非阻塞文件锁防止并发。stop 在三库 BEGIN IMMEDIATE 门闩内二次排空，
拒绝活跃 Run、未结算 execution、待处理预约及 held/unknown_pending 预算。
历史缺失绑定只能按逐条审计的三元组和 NULL scope/budget 条件保留；
Session 物理 open 不等于活跃训练。门闩只在有限优雅退出期间持有，finally
释放；超时保留身份记录和可诊断进程，不清 PID、不强杀。

每次发信号前重新核对四项身份。现场 Python 不提供 pidfd_open，使用现有
Linux /proc 身份检查边界；不承诺抵御恶意同 UID 进程。退出瞬间 cwd 权限
异常最多复查 150 ms，只有 stat 消失或僵尸状态才确认退出，身份变化拒绝，
持续权限异常抛出。Web 和已登记/当前 worker 都退出后才报告 stopped。

start 幂等：运行中必须同 release 且健康，不能静默切换代码；孤立 worker
阻断新启动。未知监听者立即阻断，不能终止它。无监听但暂不可绑定时有限
等待，SO_REUSEADDR 与 uvicorn 一致。readiness 必须是 server 模式、兼容
training-worker-guard-v1 且只有一个 live worker。失败保留已启动进程身份，
先 status 再按相同安全入口 stop。timeout 参数是上界，不是强杀许可。

发布先核对远端并验收独立分支，再取得最新一致性三库及文件备份并恢复检查。
排空停止后只切 current；启动验收失败时仍用此控制器停止，切回旧代码，
保留当前 storage 和发布后新数据。禁止用旧数据库覆盖当前生产。

Linux 验证命令：

```bash
python -m pytest backend/tests/test_release_controller.py -q
python -m pytest backend/tests agent_poc/tests -q
python -m compileall backend/app agent_poc scripts/service_control.py -q
```

故障注入限隔离实例。没有新增系统重启自启动、systemd/cgroup 或 Qwen 常驻。
