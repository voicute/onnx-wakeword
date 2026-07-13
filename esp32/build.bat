@echo off
chcp 65001 >nul
REM ============================================================
REM  Voicute Wake Word — 编译脚本
REM  测试平台: ESP32-S3-HMI-DevKit, ESP-IDF v6.0.1, Windows 11
REM
REM  前置要求:
REM    1. 安装 ESP-IDF v6.0 (https://docs.espressif.com)
REM    2. 把 .tflite 模型放到 spiffs_content/ 目录
REM    3. 把训练导出的 head.h 替换 main/head.h
REM
REM  如果 ESP-IDF 路径不同, 请修改下面的 IDF_PATH
REM ============================================================

set "IDF_PATH=C:\esp\v6.0.1\esp-idf"
set "IDF_PYTHON_ENV_PATH=%USERPROFILE%\.espressif\python_env\idf6.0_py3.10_env"

set "TOOLS_BASE=%USERPROFILE%\.espressif"
set "PATH=%TOOLS_BASE%\python_env\idf6.0_py3.10_env\Scripts;%PATH%"
set "PATH=%TOOLS_BASE%\tools\cmake\4.0.3\bin;%PATH%"
set "PATH=%TOOLS_BASE%\tools\ninja\1.12.1;%PATH%"
set "PATH=%TOOLS_BASE%\tools\xtensa-esp-elf\esp-15.2.0_20251204\xtensa-esp-elf\bin;%PATH%"
set "PATH=%TOOLS_BASE%\tools\ccache\4.12.1;%PATH%"

set http_proxy=
set https_proxy=
set no_proxy=*
set IDF_COMPONENT_VERIFY_SSL=0

cd /d "%~dp0"

echo.
echo [Voicute] 设置芯片型号 ESP32-S3...
echo.
python "%IDF_PATH%\tools\idf.py" set-target esp32s3
if %ERRORLEVEL% NEQ 0 (
    echo [Voicute] set-target 失败!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [Voicute] 开始编译...
echo.
python "%IDF_PATH%\tools\idf.py" build
if %ERRORLEVEL% NEQ 0 (
    echo [Voicute] 编译失败!
    pause
    exit /b %ERRORLEVEL%
)
echo [Voicute] 编译成功! 运行 flash.bat 烧录
