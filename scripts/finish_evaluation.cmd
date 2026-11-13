@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0finish_evaluation.ps1" %*
exit /b %ERRORLEVEL%
