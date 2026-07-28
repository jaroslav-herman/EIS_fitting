@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo Could not find uv on PATH.
    echo Open PowerShell and run: uv run python main.py
    pause
    exit /b 1
)

uv run python main.py

if errorlevel 1 (
    echo.
    echo EIS Fitting failed to start. The error is shown above.
    pause
)
endlocal
