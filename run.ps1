# PrognosAI-X Workstation Launcher Script

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "             PrognosAI-X Platform Launcher" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Check Virtual Environment
if (Test-Path ".\.venv") {
    Write-Host "[INFO] Local Python Virtual Environment detected." -ForegroundColor Green
} else {
    Write-Host "[ERROR] Virtual Environment .venv not found. Please create it first." -ForegroundColor Red
    Exit 1
}

# 2. Activate environment and run Streamlit
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"

Write-Host "[INFO] Launching PrognosAI-X Streamlit Medical Workstation..." -ForegroundColor Green
Write-Host "[INFO] Press Ctrl+C in this terminal to stop the workstation server." -ForegroundColor Yellow

& ".\.venv\Scripts\python.exe" -m streamlit run app/dashboard.py
