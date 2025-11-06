# Makefile for RAG Service
# Usage: make <target>
# Matches install.bat installation process

.PHONY: help check-python venv install verify run test ingest clean health

# Detect Python command (python3 on Linux/macOS, python on Windows)
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# Windows detection
ifeq ($(OS),Windows_NT)
	VENV_PYTHON := $(VENV)/Scripts/python.exe
	VENV_PIP := $(VENV)/Scripts/pip.exe
	VENV_ACTIVATE := $(VENV)/Scripts/activate
else
	VENV_ACTIVATE := $(VENV)/bin/activate
endif

# Default target
help:
	@echo "RAG Service Makefile Commands:"
	@echo ""
	@echo "  make venv          - Create virtual environment (.venv)"
	@echo "  make install       - Install all dependencies (creates venv if needed)"
	@echo "  make verify        - Verify installation"
	@echo "  make run           - Start RAG server"
	@echo "  make test          - Run tests"
	@echo "  make health        - Check server health"
	@echo "  make ingest        - Upload test document (requires server running)"
	@echo "  make clean         - Clean temporary files"
	@echo ""

# Check Python version
check-python:
	@echo "[1/8] Verifying Python version..."
	@$(PYTHON) --version || (echo "ERROR: Python is not installed or not in PATH!" && exit 1)
	@echo ""

# Create virtual environment if it doesn't exist
venv: check-python
	@$(PYTHON) -c "import os, sys; sys.exit(0 if os.path.exists('$(VENV_ACTIVATE)') else 1)" 2>/dev/null && \
		echo "[2/8] Virtual environment already exists, skipping creation..." || \
		(echo "[2/8] Creating virtual environment..." && \
		$(PYTHON) -m venv $(VENV) && \
		echo "✅ Virtual environment created!")
	@echo ""

# Install dependencies (matches install.bat order)
install: venv
	@echo "[3/8] Upgrading pip, setuptools, wheel..."
	@$(VENV_PIP) install --upgrade pip setuptools wheel || (echo "ERROR: Failed to upgrade pip!" && exit 1)
	@echo ""
	@echo "[4/8] Installing raganything[text]==1.2.8..."
	@echo "This may take several minutes..."
	@$(VENV_PIP) install raganything[text]==1.2.8 || (echo "ERROR: Failed to install raganything!" && exit 1)
	@echo ""
	@echo "[5/8] Installing CUDA 12.1 dependencies from requirements-torch-cuda121.txt..."
	@echo "This may take 5-10 minutes..."
	@$(VENV_PIP) install -r requirements-torch-cuda121.txt || (echo "ERROR: Failed to install CUDA dependencies!" && exit 1)
	@echo ""
	@echo "[6/8] Installing other dependencies from requirements.txt..."
	@echo "This may take 5-10 minutes..."
	@$(VENV_PIP) install -r requirements.txt || (echo "ERROR: Failed to install other dependencies!" && exit 1)
	@echo ""
	@echo "[7/8] Verifying installation..."
	@$(VENV_PYTHON) -c "import raganything; print('✓ raganything installed')" || (echo "WARNING: raganything verification failed!" && exit 1)
	@$(VENV_PYTHON) -c "import peft; print('✓ peft installed')" || (echo "WARNING: peft verification failed!" && exit 1)
	@$(VENV_PYTHON) -c "import chromadb; print('✓ chromadb installed')" || (echo "WARNING: chromadb verification failed!" && exit 1)
	@$(VENV_PYTHON) -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}')" || (echo "WARNING: PyTorch verification failed!" && exit 1)
	@$(VENV_PYTHON) -c "import fastapi, transformers; print('✓ Core packages OK!')" || (echo "WARNING: Core packages verification failed!" && exit 1)
	@echo "All core packages verified successfully!"
	@echo ""
	@echo "[8/8] Running deployment validation..."
	@$(VENV_PYTHON) -c "import os; exit(0 if os.path.exists('scripts/validate_deployment.py') else 1)" 2>/dev/null && \
		$(VENV_PYTHON) scripts/validate_deployment.py || \
		echo "WARNING: Validation script not found, skipping..."
	@echo ""
	@echo "========================================"
	@echo "Installation Complete!"
	@echo "========================================"
	@echo ""
	@echo "Next steps:"
	@echo "1. Review the README.md for details"
	@echo "2. Copy .env.example to .env and configure it"
	@echo "3. Start the server: make run"
	@echo "4. Access API docs: http://localhost:8000/docs"
	@echo ""

# Verify installation
verify: venv
	@echo "Verifying installation..."
	@$(VENV_PYTHON) -c "import raganything; print('✓ raganything')"
	@$(VENV_PYTHON) -c "import peft; print('✓ peft')"
	@$(VENV_PYTHON) -c "import chromadb; print('✓ chromadb')"
	@$(VENV_PYTHON) -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}')"
	@$(VENV_PYTHON) -c "import fastapi, transformers; print('✓ Core packages OK!')"
	@echo "✅ Verification complete"

# Run server (uses venv Python)
run: venv
	@echo "Starting RAG server..."
	@$(VENV_PYTHON) scripts/run_server.py

# Run tests (uses venv Python)
test: venv
	@echo "Running tests..."
	@$(VENV_PYTHON) -m pytest tests/ -v

# Check server health
health:
	@echo "Checking server health..."
	@curl -s http://localhost:8000/api/v1/health | $(PYTHON) -m json.tool || echo "❌ Server not running. Start with: make run"

# Upload test document (requires server running)
ingest:
	@echo "Uploading test document..."
	@$(PYTHON) -c "import os; os.makedirs('data/uploads', exist_ok=True)" 2>/dev/null || true
	@$(PYTHON) -c "import os; exit(0 if os.path.exists('data/uploads/test_lora.txt') else 1)" 2>/dev/null && \
		echo "Test document exists" || \
		(echo "Creating test document..." && \
		$(PYTHON) -c "with open('data/uploads/test_lora.txt', 'w') as f: f.write('LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds trainable low-rank matrices to pre-trained models.')")
	@curl -X POST http://localhost:8000/api/v1/rag/documents \
		-H "Content-Type: application/json" \
		-d '{"file_path": "./data/uploads/test_lora.txt", "collection_name": "default"}' \
		| $(PYTHON) -m json.tool || echo "❌ Upload failed. Is server running?"

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@echo "✅ Cleaned"



