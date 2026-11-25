@echo off
chcp 65001 >nul
title Tao file cai dat Net Anh
cd /d "%~dp0"
rem Tao MOT file cai dat NetAnh-PHIENBAN-setup.exe ngay trong thu muc nay (tu cai Python, Inno Setup neu thieu).
rem Xem HUONG_DAN_BUILD.md. Tham so them: -Flavor cuda, -NoModels, -NoInstaller, -OutDir THU_MUC
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\build_windows.ps1" %*
pause
