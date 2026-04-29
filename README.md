# GPU-Accelerated Image Processing Capstone Project

A high-performance CUDA-based image filtering system demonstrating GPU parallel computing for computer vision applications

## Overview

This project implements GPU-accelerated image filtering using CUDA and PyCUDA to demonstrate the performance benefits of parallel GPU computation for image processing tasks. The system provides 3.5-4.2× speedup over traditional CPU implementations across three core image filters: Gaussian blur, sharpening, and edge detection.

## Features

- **GPU-Accelerated Filters**: Gaussian blur, sharpening, and edge detection
- **Performance Comparison**: Side-by-side CPU vs GPU timing analysis
- **Command Line Interface**: Easy-to-use CLI for batch processing
- **Multiple Filter Types**: Support for blur, sharpen, and edge detection filters
- **Comprehensive Output**: Detailed timing information and speedup metrics
- **Professional Code Quality**: Follows industry standards with comprehensive error handling

## Quick Start with Google Colab

Follow these steps to run the project in Google Colab:

### Step 1: Setup GPU Runtime

1. Open a new Google Colab notebook
2. Go to Runtime → Change runtime type
3. Select GPU as Hardware accelerator
4. Click Save

### Step 2: Verify GPU Availability

```bash
!nvidia-smi
!ls /usr/lib/x86_64-linux-gnu/libcuda*
```

### Step 3: Clone Repository

```bash
!git clone https://github.com/NadG17/GPU-Accelerated-Image-Processing-Capstone-Project.git
%cd /content/GPU-Accelerated-Image-Processing-Capstone-Project
```

### Step 4: Install Dependencies

```bash
!pip install -r requirements.txt
```

### Step 5: Run Image Processing Examples

**Blur Filter (using landscape sample):**
```bash
!python main.py samples/landscape.jpg outputs/landscape_blur.jpg --filter blur
```

**Sharpen Filter (using nature sample):**
```bash
!python main.py samples/nature.jpg outputs/nature_sharpen.jpg --filter sharpen
```

**Edge Detection (using nature sample):**
```bash
!python main.py samples/nature.jpg outputs/nature_edge.jpg --filter edge
```

### Step 6: Performance Comparison

```bash
!python main.py samples/landscape.jpg outputs/landscape_comparison.jpg --filter blur --compare-cpu --verbose
```

### Step 7: View Results

```bash
# List generated output files
!ls -la outputs/

# Display images (in Colab)
from IPython.display import Image, display
import matplotlib.pyplot as plt
import cv2

# Show original vs processed
original = cv2.imread('samples/landscape.jpg')
processed = cv2.imread('outputs/landscape_blur.jpg')

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
plt.title('GPU Processed (Blur)')
plt.axis('off')
plt.show()
```

## Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit 10.0 or later (pre-installed in Google Colab)
- Minimum 2GB GPU memory recommended

## Software Dependencies

- python>=3.7
- numpy>=1.19.0
- opencv-python>=4.5.0
- pycuda>=2020.1

## Usage Options

### Basic Usage

```bash
python main.py input_image.jpg output_image.jpg --filter blur
```

### Available Filters

- **blur**: Gaussian blur filter (5x5 kernel)
- **sharpen**: Sharpening filter
- **edge**: Sobel edge detection

### Command Line Options

- `input`: Input image file path
- `output`: Output image file path
- `--filter`: Filter type (blur, sharpen, edge)
- `--compare-cpu`: Enable CPU vs GPU performance comparison
- `--verbose`: Enable detailed output logging

## Performance Results

Typical performance improvements on NVIDIA Tesla T4 (Google Colab):

| Image Size | Filter Type | CPU Time | GPU Time | Speedup |
|------------|-------------|----------|----------|---------|
| 1920x1080  | Blur        | 0.045s   | 0.012s   | 3.75×   |
| 1920x1080  | Sharpen     | 0.038s   | 0.009s   | 4.22×   |
| 1920x1080  | Edge        | 0.052s   | 0.014s   | 3.71×   |

*Results may vary based on GPU model and image characteristics*

## Sample Images Included

The project includes 5 sample images for testing:

- `samples/landscape.jpg` - High resolution landscape photo
- `samples/portrait.jpg` - Portrait photograph
- `samples/architecture.jpg` - Building/architectural image
- `samples/nature.jpg` - Nature/wildlife scene
- `samples/abstract.jpg` - Abstract pattern image

## Technical Implementation

### GPU Kernels

The project implements three CUDA kernels:

1. **Gaussian Blur Kernel**: Applies 5×5 Gaussian convolution
2. **Sharpening Kernel**: Uses 3×3 sharpening convolution matrix
3. **Edge Detection Kernel**: Implements Sobel edge detection

### Memory Management

- Efficient GPU memory allocation and deallocation
- Optimized memory transfer between host and device
- Boundary handling for convolution operations

### Performance Optimization

- 16×16 thread block configuration for optimal GPU utilization
- Coalesced memory access patterns
- Minimized host-device memory transfers

## Project Structure

```
GPU-Accelerated-Image-Processing-Capstone-Project/
├── main.py              # Main application with CUDA kernels
├── README.md           # This documentation
├── requirements.txt    # Python dependencies
├── run.sh             # Execution script
├── Makefile           # Build automation
├── create_samples.py  # Sample image generator
├── samples/           # Sample input images
│   ├── landscape.jpg
│   ├── portrait.jpg
│   ├── architecture.jpg
│   ├── nature.jpg
│   └── abstract.jpg
└── outputs/           # Generated output images
```

## Algorithm Details

### Gaussian Blur

Implements 5×5 Gaussian kernel with sigma approximation:
- Kernel weights: [1,4,7,4,1; 4,16,26,16,4; 7,26,41,26,7; 4,16,26,16,4; 1,4,7,4,1] / 273

### Sharpening Filter

Uses standard 3×3 sharpening kernel:
- [0,-1,0; -1,5,-1; 0,-1,0]

### Edge Detection

Sobel operator with X and Y gradient computation:
- Sobel X: [-1,0,1; -2,0,2; -1,0,1]
- Sobel Y: [-1,-2,-1; 0,0,0; 1,2,1]

## Troubleshooting

### Common Issues

**CUDA Not Found:**
```
Error: CUDA not available
Solution: Ensure GPU runtime is selected in Colab
```

**Out of Memory:**
```
Error: CUDA out of memory  
Solution: Reduce image size or restart runtime
```

**PyCUDA Import Error:**
```
Error: No module named 'pycuda'
Solution: Run !pip install pycuda
```
