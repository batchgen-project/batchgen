# ---------------------------------------------------------------------------- #
#  BatchGen Makefile                                                           #
#  Copyright (c) EfficientMoE team 2025                                        #
# ---------------------------------------------------------------------------- #

.PHONY: help install install-all install-deps install-hopper-deps install-kernels clean

# Default target
help:
	@echo "BatchGen Installation Targets"
	@echo ""
	@echo "  make install              Install batchgen_kernels + BatchGen (assumes deps are installed)"
	@echo "  make install-kernels      Install batchgen_kernels CUDA extensions only"
	@echo "  make install-all          Install BatchGen with all Hopper dependencies"
	@echo "  make install-deps         Install Hopper dependencies only (flash-attn, FlashMLA, DeepGEMM)"
	@echo ""
	@echo "Individual dependency targets:"
	@echo "  make install-flash-attn   Install flash-attention 3 (Hopper)"
	@echo "  make install-flashmla     Install FlashMLA"
	@echo "  make install-deepgemm     Install DeepGEMM"
	@echo ""
	@echo "Other targets:"
	@echo "  make clean                Clean build artifacts"
	@echo "  make help                 Show this help message"

# Install BatchGen and its CUDA kernel extensions
install: install-kernels
	pip install .

# Install CUDA kernel extensions only
install-kernels:
	cd batchgen_kernels && pip install . --no-build-isolation

# Install all dependencies + BatchGen
install-all:
	./scripts/install_deps.sh --all

# Install Hopper dependencies only
install-deps:
	./scripts/install_deps.sh --flash-attn --flashmla --deepgemm

# Individual dependency targets
install-flash-attn:
	./scripts/install_deps.sh --flash-attn

install-flashmla:
	./scripts/install_deps.sh --flashmla

install-deepgemm:
	./scripts/install_deps.sh --deepgemm

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.so" -delete 2>/dev/null || true
