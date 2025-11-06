#!/bin/bash
# Automated installation script for Linux/Mac
# This script creates venv and installs all dependencies in the correct order

echo "========================================"
echo "RAFT-RAG Server Installation"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH!"
    echo "Please install Python 3.11+ from https://www.python.org/downloads/"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Verify Python version
echo "[1/8] Verifying Python version..."
$PYTHON_CMD --version
echo ""

# Create virtual environment if it doesn't exist
if [ ! -f ".venv/bin/activate" ]; then
    echo "[2/8] Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment!"
        exit 1
    fi
    echo "Virtual environment created successfully!"
else
    echo "[2/8] Virtual environment already exists, skipping creation..."
fi

# Activate virtual environment
echo "[3/8] Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment!"
    exit 1
fi

# Upgrade pip
echo "[4/8] Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upgrade pip!"
    exit 1
fi
echo ""

# Install raganything FIRST (installs most dependencies)
echo "[5/8] Installing raganything[text]==1.2.8..."
echo "This may take several minutes..."
pip install raganything[text]==1.2.8
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install raganything!"
    exit 1
fi
echo ""

# Install cuda121 dependencies
echo "[6/8] Installing cuda121 dependencies from requirements-torch-cuda121.txt..."
echo "This may take 5-10 minutes..."
pip install -r requirements-torch-cuda121.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install cuda121 requirements!"
    exit 1
fi
echo ""

# Install other dependencies from requirements.txt
echo "[7/8] Installing other dependencies from requirements.txt..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install other dependencies!"
    exit 1
fi
echo ""

# Verify installation
echo "[8/8] Verifying installation..."
echo "========================================"
python -c "import raganything; print('✓ raganything installed')" || echo "WARNING: raganything verification failed"
python -c "import peft; print('✓ peft installed')" || echo "WARNING: peft verification failed"
python -c "import chromadb; print('✓ chromadb installed')" || echo "WARNING: chromadb verification failed"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}')" || echo "WARNING: torch verification failed"
python -c "import fastapi, transformers; print('✓ Core packages OK!')" || echo "WARNING: Core packages verification failed"
echo ""

# Check if all verifications passed
if python -c "import raganything, peft, chromadb, torch, fastapi, transformers" 2>/dev/null; then
    echo "All core packages verified successfully!"
else
    echo "WARNING: Some packages failed verification!"
    echo "Please check the installation manually."
fi
echo ""

# Run validation script
echo "========================================"
echo "Running deployment validation..."
echo "========================================"
if [ -f "scripts/validate_deployment.py" ]; then
    python scripts/validate_deployment.py
    echo ""
else
    echo "WARNING: Validation script not found, skipping..."
    echo ""
fi

echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review the README.md for details"
echo "2. Copy .env.example to .env and configure it"
echo "3. Start the server: python scripts/run_server.py"
echo "4. Access API docs: http://localhost:8000/docs"
echo ""
