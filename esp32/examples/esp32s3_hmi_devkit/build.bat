@echo off
REM ============================================================
REM  Voicute Wake Word - Build Script
REM  Platform: ESP32-S3-HMI-DevKit, ESP-IDF v6.0.1, Windows 11
REM
REM  Prerequisites:
REM    1. Install ESP-IDF v6.0 (https://docs.espressif.com)
REM    2. Put .tflite model in spiffs_content/
REM    3. Replace main/head.h with your own weights
REM
REM  Change IDF_PATH below if your ESP-IDF is elsewhere
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
echo [Voicute] Setting target ESP32-S3...
echo.
python "%IDF_PATH%\tools\idf.py" set-target esp32s3
if %ERRORLEVEL% NEQ 0 (
    echo [Voicute] set-target FAILED!
    pause
    exit /b %ERRORLEVEL%
)

echo [Voicute] Patching led_strip component...
python patch_led_strip.py

echo.
echo [Voicute] Building...
echo.
python "%IDF_PATH%\tools\idf.py" build
if %ERRORLEVEL% NEQ 0 (
    echo [Voicute] Build FAILED!
    pause
    exit /b %ERRORLEVEL%
)
echo [Voicute] Build OK! Run flash.bat to flash
