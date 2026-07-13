@echo off
chcp 65001 >nul
REM ============================================================
REM  Voicute Wake Word — 烧录 + 串口监控
REM  测试平台: ESP32-S3-HMI-DevKit, Windows 11
REM
REM  修改 COM5 为你的串口号
REM ============================================================

set "IDF_PATH=C:\esp\v6.0.1\esp-idf"
set "TOOLS_BASE=%USERPROFILE%\.espressif"
set "PATH=%TOOLS_BASE%\python_env\idf6.0_py3.10_env\Scripts;%PATH%"
set "PATH=%TOOLS_BASE%\tools\cmake\4.0.3\bin;%PATH%"
set "PATH=%TOOLS_BASE%\tools\ninja\1.12.1;%PATH%"
set "PATH=%TOOLS_BASE%\tools\xtensa-esp-elf\esp-15.2.0_20251204\xtensa-esp-elf\bin;%PATH%"

cd /d "%~dp0"

echo [Voicute] 烧录 app + 模型...
python -m esptool --chip esp32s3 -p COM5 -b 460800 --before default-reset --after hard-reset write-flash 0x10000 build\factory_01.bin 0x957000 build\models.bin
if %ERRORLEVEL% NEQ 0 (
    echo [Voicute] 烧录失败! 检查 COM 口和设备连接
    pause
    exit /b 1
)

echo [Voicute] 烧录成功! 启动串口监控...
echo 按 Ctrl+] 退出监控
python -m esp_idf_monitor --port COM5
pause
