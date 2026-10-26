@echo off
rem One command to start ResearchLens on Windows without Docker -- see start-dev.ps1.
rem Bypasses PowerShell's script policy for this script only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-dev.ps1" %*
