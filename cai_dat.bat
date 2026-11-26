@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo === Cai dat Net Anh ===
where python >nul 2>nul || (echo Chua co Python. Cai Python 3.11 hoac 3.12 tu python.org, nho tick "Add to PATH". & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  set "CUIDX=cu128"
  for /f "tokens=1 delims=." %%c in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
    if %%c LSS 7 set "CUIDX=cu126"
  )
  call echo Tim thay GPU NVIDIA: cai PyTorch ban CUDA %%CUIDX%% - cu126 cho GTX 10 va GTX 900...
  call pip install torch torchvision --index-url https://download.pytorch.org/whl/%%CUIDX%% || pip install torch torchvision
) else (
  echo Khong thay GPU NVIDIA: cai PyTorch ban CPU.
  pip install torch torchvision
)
pip install -r requirements.txt
echo.
echo Xong. Chay phan mem bang file chay.bat
pause
