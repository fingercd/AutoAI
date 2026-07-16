param(
  [int]$LocalPort = 8000,
  [int]$RemotePort = 8000
)

# The remote service binds loopback in the recommended deployment, so expose it only
# through an SSH local-forward instead of opening the cluster port publicly.
Write-Host "Opening SSH tunnel: http://127.0.0.1:$LocalPort -> node3:$RemotePort"
ssh -N -L "$LocalPort`:127.0.0.1:$RemotePort" node3
