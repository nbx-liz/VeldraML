@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0start_gui.ps1" %*
endlocal
