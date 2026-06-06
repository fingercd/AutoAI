# AutoAI

本地开发启动命令：

```powershell
& "$env:USERPROFILE\anaconda3\envs\pytorch\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

当前第一版支持：

- 加载项目根目录 `data.csv` 并预览样本、类别分布和曲线。
- 上传统一格式建模 CSV。
- 使用 PyTorch 1D-CNN 启动本地训练。
- 查看训练状态、指标和下载 `predictions.csv` / `metrics.json` / `model.pt`。
- 上传拉曼或色谱原始 CSV，并按行范围整理为统一格式 CSV。
