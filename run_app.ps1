$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 1. Stop any leftover Streamlit / Python server holding port 8512
$busy = Get-NetTCPConnection -LocalPort 8512 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $busy) {
    if ($procId -and $procId -ne $PID) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}

# 2. Pick a free port starting from 8512
$port = 8512
while (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    $port++
}

# 3. Run Streamlit consistently via uv
Write-Host "Starting Fish Freshness App on port $port..." -ForegroundColor Green
uv run streamlit run app.py --server.port $port --server.headless true