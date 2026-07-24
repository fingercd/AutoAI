# 【中文说明】本机（Windows/PowerShell）到集群 node3 的 SSH 本地端口转发脚本。
#
# 背景：推荐部署中远程服务只绑定回环地址（127.0.0.1），集群端口不对公网开放；
# 因此通过 SSH local-forward 把远端的 127.0.0.1:RemotePort 映射到本机
# 127.0.0.1:LocalPort，浏览器访问本机地址即可安全使用平台。
#
# 用法：
#   ./local_tunnel.ps1                      # 默认 8000 -> 8000
#   ./local_tunnel.ps1 -LocalPort 9000      # 自定义本机端口
param(
  [int]$LocalPort = 8000,
  [int]$RemotePort = 8000
)

# The remote service binds loopback in the recommended deployment, so expose it only
# through an SSH local-forward instead of opening the cluster port publicly.
Write-Host "Opening SSH tunnel: http://127.0.0.1:$LocalPort -> node3:$RemotePort"
# ssh 参数说明：-N 不执行远程命令（纯转发）；-L 建立本地端口转发；
# "LocalPort:127.0.0.1:RemotePort" 中目标侧的 127.0.0.1 是远端视角的回环地址。
# 反引号 ` 是 PowerShell 转义符，`: 用于把变量名与紧跟的冒号分隔开。
# node3 需在 ~/.ssh/config 中预先配置好主机别名；隧道建立后终端会一直挂起，
# 按 Ctrl+C 即可关闭隧道。
ssh -N -L "$LocalPort`:127.0.0.1:$RemotePort" node3
