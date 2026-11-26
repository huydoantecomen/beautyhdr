@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
rem Beauty HDR ban nha phat trien: chay tu ma nguon (co menu Nha phat trien, xuat prompt cho AI ngoai).
cd /d "%~dp0ma-nguon"
if not exist .venv\Scripts\python.exe (
  echo Lan dau: cai moi truong Python cho ban dev...
  call cai_dat.bat
)
if not exist .venv\Scripts\python.exe (echo Cai dat chua xong. & pause & exit /b 1)
start "" .venv\Scripts\pythonw.exe -m net_anh_dev %*
