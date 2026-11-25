@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Cai dat Net Anh ===
where python >nul 2>nul || (echo Chua co Python. Cai Python 3.11 hoac 3.12 tu python.org, nho tick "Add to PATH". & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo Tim thay GPU NVIDIA: cai PyTorch ban CUDA...
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || pip install torch torchvision
) else (
  echo Khong thay GPU NVIDIA: cai PyTorch ban CPU.
  pip install torch torchvision
)
pip install -r requirements.txt
echo.
echo Xong. Chay phan mem bang file chay.bat
pause
