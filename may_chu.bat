@echo off
rem May chu cho app Net Anh tren dien thoai: cua so hien dia chi va ma ghep noi. Dong cua so la dung.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" pythonw -m net_anh.server_gui %*
