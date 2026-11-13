@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0continue_project.ps1" %*
exit /b %ERRORLEVEL%
