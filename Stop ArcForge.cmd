@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Stop ArcForge.ps1"
if errorlevel 1 pause
