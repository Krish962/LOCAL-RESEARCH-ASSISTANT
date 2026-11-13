@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0validate_40.ps1" %*
exit /b %ERRORLEVEL%
