@echo off
chcp 65001 >nul
title Chuyen Beauty HDR sang ban GPU NVIDIA
if not exist "%~dp0ma-nguon\packaging\cap_nhat.ps1" (
  echo Thieu thu muc ma-nguon canh file nay.
  echo Hay giai nen CA goi cap nhat ra mot cho bat ky roi bam dup lai file nay.
  pause
  exit /b 1
)
echo Chuyen sang ban GPU NVIDIA (can mang, tai them khoang 3 GB thu vien GPU mot lan). KHONG tai lai model.
echo Se hien hop chon thu muc goc cua Beauty HDR ban cu. Dong Beauty HDR neu dang mo.
powershell -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0ma-nguon\packaging\cap_nhat.ps1" -Flavor cuda %*
pause
