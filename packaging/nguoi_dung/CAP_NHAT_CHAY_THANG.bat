@echo off
chcp 65001 >nul
title Cap nhat Beauty HDR (chay thang, khong cai setup.exe)
if not exist "%~dp0ma-nguon\packaging\cap_nhat.ps1" (
  echo Thieu thu muc ma-nguon canh file nay.
  echo Hay giai nen CA goi cap nhat ra mot cho bat ky roi bam dup lai file nay.
  pause
  exit /b 1
)
echo Cap nhat Beauty HDR chay thang tu ma nguon (.venv, chay.bat). KHONG dong goi, KHONG cai setup.exe.
echo Se hien hop chon thu muc goc cua Beauty HDR ban cu (thu muc co ma-nguon\.venv). Dong Beauty HDR neu dang mo.
powershell -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0ma-nguon\packaging\cap_nhat.ps1" -Mode chay %*
pause
