"""Agent Session / Experiment / Feedback 包的命名空间标记。

本目录的实现负责把 Session/Experiment 的状态写入 ``storage/agent.sqlite3``，并
为 ``routers/agent.py`` 提供 service/repository 装配。整个包刻意保持零业务逻辑
依赖：训练服务 ``backend.app.training`` 与 ``RunRepository`` 都从父包导入，避免
形成 ``agent ↔ runs ↔ agent`` 的循环引用。
"""