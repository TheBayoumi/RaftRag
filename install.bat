@echo off
REM Automated installation script for Windows
REM This script creates venv and installs all dependencies in the correct order

echo ========================================
echo RAFT-RAG Server Installation
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    exit /b 1
)

REM Verify Python version
echo [1/7] Verifying Python version...
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo [2/7] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo [2/7] Virtual environment already exists, skipping creation...
)

REM Activate virtual environment
echo [3/7] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    exit /b 1
)

REM Upgrade pip
echo [4/7] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip!
    exit /b 1
)
echo.

REM Install raganything FIRST (installs most dependencies)
echo [5/7] Installing raganything[text]==1.2.8...
echo This may take several minutes...
pip install raganything[text]==1.2.8
if errorlevel 1 (
    echo ERROR: Failed to install raganything!
    exit /b 1
)
echo.

REM Install cuda121 dependencies
echo [6/7] Installing cuda121 dependencies from requirements-torch-cuda121.txt...
echo This may take 5-10 minutes...
pip install -r requirements-torch-cuda121.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements!
    exit /b 1
)
echo.

REM Install other dependencies from requirements.txt
echo [7/7] Installing other dependencies from requirements.txt...
echo This may take 5-10 minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install other dependencies!
    exit /b 1
)
echo.

REM Verify installation
echo [8/8] Verifying installation...
echo ========================================
python -c "import raganything; print('✓ raganything installed')"
python -c "import peft; print('✓ peft installed')"
python -c "import chromadb; print('✓ chromadb installed')"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}')"
python -c "import fastapi, transformers; print('✓ Core packages OK!')"
if errorlevel 1 (
    echo WARNING: Some packages failed verification!
    echo Please check the installation manually.
) else (
    echo All core packages verified successfully!
)
echo.

REM Run validation script
echo ========================================
echo Running deployment validation...
echo ========================================
if exist "scripts\validate_deployment.py" (
    python scripts\validate_deployment.py
    echo.
) else (
    echo WARNING: Validation script not found, skipping...
    echo.
)

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Review the README.md for details
echo 2. Copy .env.example to .env and configure it
echo 3. Start the server: python scripts\run_server.py
echo 4. Access API docs: http://localhost:8000/docs
echo.
pause
