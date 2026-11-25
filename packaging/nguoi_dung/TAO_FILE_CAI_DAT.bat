@echo off
chcp 65001 >nul
title Tao file cai dat Net Anh
cd /d "%~dp0"
if not exist "%~dp0ma-nguon\packaging\build_windows.ps1" (
  echo Khong thay thu muc ma-nguon canh file nay.
  echo Hay giai nen CA file zip ra mot thu muc roi bam dup lai file nay.
  pause
  exit /b 1
)
rem Tao MOT file NetAnh-PHIENBAN-setup.exe ngay canh file nay. Chi tiet: ma-nguon\HUONG_DAN_BUILD.md
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ma-nguon\packaging\build_windows.ps1" -OutDir "%~dp0." %*
pause
