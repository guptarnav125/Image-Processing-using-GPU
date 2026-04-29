# GPU-Accelerated Image Processing Makefile
# Author: GPU Programming Specialization Student

PYTHON := python3
PIP := pip3
MAIN_SCRIPT := main.py
SAMPLES_DIR := samples
OUTPUTS_DIR := outputs

.PHONY: all setup test clean run demo help install-deps check-cuda

# Default target
all: setup demo

# Help target
help:
	@echo "GPU-Accelerated Image Processing - Makefile"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Setup and run demo (default)"
	@echo "  setup       - Install dependencies and create directories"
	@echo "  install-deps - Install Python dependencies"
	@echo "  check-cuda  - Check CUDA availability"
	@echo "  test        - Run basic functionality test"
	@echo "  demo        - Run demonstration with sample images"
	@echo "  run         - Run with custom parameters (use ARGS=...)"
	@echo "  clean       - Clean up generated files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                                    # Run full demo"
	@echo "  make run ARGS=\"input.jpg output.jpg --filter blur\""
	@echo "  make test"
	@echo "  make clean"

# Setup target - prepare environment
setup: install-deps check-cuda
	@echo "Setting up project directories..."
	@mkdir -p $(SAMPLES_DIR)
	@mkdir -p $(OUTPUTS_DIR)
	@echo "Setup complete!"

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "Dependencies installed!"

# Check CUDA availability
check-cuda:
	@echo "Checking CUDA availability..."
	@$(PYTHON) -c "import pycuda.autoinit; print('✓ CUDA is working correctly')" || \
		(echo "✗ CUDA not available. Please install NVIDIA drivers and CUDA toolkit."; exit 1)

# Test basic functionality
test: setup
	@echo "Running basic functionality test..."
	@echo "This test requires at least one sample image in $(SAMPLES_DIR)/"
	@if [ ! -f "$(SAMPLES_DIR)/test.jpg" ] && [ ! -f "$(SAMPLES_DIR)/landscape.jpg" ]; then \
		echo "Warning: No test images found. Please add images to $(SAMPLES_DIR)/ directory"; \
		echo "Test cannot proceed without sample images."; \
		exit 1; \
	fi
	@# Find first available image for testing
	@TEST_IMAGE=$$(ls $(SAMPLES_DIR)/*.jpg 2>/dev/null | head -1); \
	if [ -n "$$TEST_IMAGE" ]; then \
		echo "Testing with image: $$TEST_IMAGE"; \
		$(PYTHON) $(MAIN_SCRIPT) "$$TEST_IMAGE" "$(OUTPUTS_DIR)/test_output.jpg" \
			--filter blur --verbose; \
		if [ $$? -eq 0 ]; then \
			echo "✓ Basic test passed!"; \
		else \
			echo "✗ Basic test failed!"; \
			exit 1; \
		fi; \
	else \
		echo "No test images available"; \
		exit 1; \
	fi

# Run demonstration
demo: setup
	@echo "Running GPU Image Processing demonstration..."
	@./run.sh

# Run with custom arguments
run: setup
	@if [ -z "$(ARGS)" ]; then \
		echo "Usage: make run ARGS=\"input.jpg output.jpg --filter blur\""; \
		echo "Available filters: blur, sharpen, edge"; \
		echo "Additional options: --compare-cpu --verbose"; \
		exit 1; \
	fi
	@$(PYTHON) $(MAIN_SCRIPT) $(ARGS)

# Performance benchmark
benchmark: setup
	@echo "Running performance benchmark..."
	@if [ ! -f "$(SAMPLES_DIR)/benchmark.jpg" ]; then \
		echo "Creating benchmark test image..."; \
		$(PYTHON) -c "import cv2; import numpy as np; img=np.random.randint(0,255,(1080,1920,3),dtype=np.uint8); cv2.imwrite('$(SAMPLES_DIR)/benchmark.jpg', img)"; \
	fi
	@echo "Benchmarking blur filter..."
	@$(PYTHON) $(MAIN_SCRIPT) "$(SAMPLES_DIR)/benchmark.jpg" "$(OUTPUTS_DIR)/benchmark_blur.jpg" \
		--filter blur --compare-cpu --verbose
	@echo "Benchmarking sharpen filter..."
	@$(PYTHON) $(MAIN_SCRIPT) "$(SAMPLES_DIR)/benchmark.jpg" "$(OUTPUTS_DIR)/benchmark_sharpen.jpg" \
		--filter sharpen --compare-cpu --verbose
	@echo "Benchmarking edge detection filter..."
	@$(PYTHON) $(MAIN_SCRIPT) "$(SAMPLES_DIR)/benchmark.jpg" "$(OUTPUTS_DIR)/benchmark_edge.jpg" \
		--filter edge --compare-cpu --verbose
	@echo "Benchmark complete! Check outputs/ directory for results."

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	@rm -f $(OUTPUTS_DIR)/*
	@rm -f *.pyc
	@rm -rf __pycache__
	@echo "Cleanup complete!"

# Create sample images (for testing when no samples available)
create-samples:
	@echo "Creating synthetic sample images for testing..."
	@$(PYTHON) -c "
import cv2
import numpy as np
import os

# Create samples directory if it doesn't exist
os.makedirs('$(SAMPLES_DIR)', exist_ok=True)

# Generate different types of test images
samples = {
    'landscape.jpg': lambda: np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8),
    'portrait.jpg': lambda: np.random.randint(80, 220, (960, 720, 3), dtype=np.uint8),
    'architecture.jpg': lambda: np.random.randint(30, 180, (800, 1200, 3), dtype=np.uint8),
    'nature.jpg': lambda: np.random.randint(60, 240, (1080, 1920, 3), dtype=np.uint8),
    'abstract.jpg': lambda: np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
}

for filename, generator in samples.items():
    filepath = os.path.join('$(SAMPLES_DIR)', filename)
    if not os.path.exists(filepath):
        img = generator()
        cv2.imwrite(filepath, img)
        print(f'Created {filepath}')
    else:
        print(f'Sample {filepath} already exists')
"
	@echo "Sample images created!"

# Development targets
dev-setup: setup create-samples
	@echo "Development environment ready!"

# Validate code style (basic check)
lint:
	@echo "Running basic code validation..."
	@$(PYTHON) -m py_compile $(MAIN_SCRIPT)
	@echo "Code validation passed!"

# Show project statistics
stats:
	@echo "Project Statistics:"
	@echo "=================="
	@echo "Lines of code in main.py: $(wc -l < $(MAIN_SCRIPT))"
	@echo "Sample images: $(ls $(SAMPLES_DIR)/*.jpg 2>/dev/null | wc -l)"
	@echo "Output images: $(ls $(OUTPUTS_DIR)/*.jpg 2>/dev/null | wc -l)"
	@echo "Python dependencies: $(wc -l < requirements.txt)"

# Install system dependencies (Ubuntu/Debian)
install-system-deps:
	@echo "Installing system dependencies (Ubuntu/Debian)..."
	@sudo apt update
	@sudo apt install -y python3-pip python3-dev
	@sudo apt install -y nvidia-cuda-toolkit nvidia-driver-470
	@echo "System dependencies installed. Please reboot if GPU drivers were updated."
