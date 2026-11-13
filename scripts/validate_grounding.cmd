@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0validate_grounding.ps1" %*
exit /b %ERRORLEVEL%
