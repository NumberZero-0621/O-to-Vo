@echo off
echo ========================================================
echo Installing O-to-Vo dependencies...
echo ========================================================

:: .venv が存在する場合は有効化する、存在しない場合は作成する
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] .venv not found. Creating a new virtual environment with Python 3.12...
    py -3.12 -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
)

echo.
echo [1/3] Installing standard libraries...
pip install torchaudio librosa numpy scipy torch pyworld pyopenjtalk jaconv transformers g2p_en torchcrepe pykakasi
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install standard dependencies.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Installing whisperx dependencies...
pip install faster-whisper ctranslate2 pyannote-audio
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install whisperx dependencies.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/3] Installing whisperx...
pip install git+https://github.com/m-bain/whisperx.git
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install whisperx.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo All installations completed successfully!
echo ========================================================
pause
