param(
  [int]$LocalPort = 8000,
  [int]$RemotePort = 8000
)

Write-Host "Opening SSH tunnel: http://127.0.0.1:$LocalPort -> node3:$RemotePort"
ssh -N -L "$LocalPort`:127.0.0.1:$RemotePort" node3
