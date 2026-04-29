#!/usr/bin/env python3
"""
GPU-Accelerated Image Processing Capstone Project
Author: GPU Programming Specialization Student
Description: Applies various image filters using CUDA GPU acceleration
"""

import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import argparse
import time
import os
from typing import Tuple, Optional

class GPUImageProcessor:
    """GPU-accelerated image processing class using PyCUDA"""
    
    def __init__(self):
        """Initialize CUDA kernels for different image filters"""
        self.mod = SourceModule("""
        __global__ void gaussian_blur_kernel(unsigned char *input, unsigned char *output, 
                                            int width, int height, int channels) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= width || y >= height) return;
            
            // Gaussian 5x5 kernel
            float kernel[25] = {
                1, 4, 7, 4, 1,
                4, 16, 26, 16, 4,
                7, 26, 41, 26, 7,
                4, 16, 26, 16, 4,
                1, 4, 7, 4, 1
            };
            float kernel_sum = 273.0f;
            
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        // Handle boundaries by clamping
                        nx = max(0, min(width - 1, nx));
                        ny = max(0, min(height - 1, ny));
                        
                        int idx = (ny * width + nx) * channels + c;
                        int kidx = (ky + 2) * 5 + (kx + 2);
                        sum += input[idx] * kernel[kidx];
                    }
                }
                
                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = (unsigned char)(sum / kernel_sum);
            }
        }
        
        __global__ void sharpen_kernel(unsigned char *input, unsigned char *output,
                                     int width, int height, int channels) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= width || y >= height) return;
            
            // Sharpening kernel
            float kernel[9] = {
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0
            };
            
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        // Handle boundaries
                        nx = max(0, min(width - 1, nx));
                        ny = max(0, min(height - 1, ny));
                        
                        int idx = (ny * width + nx) * channels + c;
                        int kidx = (ky + 1) * 3 + (kx + 1);
                        sum += input[idx] * kernel[kidx];
                    }
                }
                
                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = (unsigned char)max(0.0f, min(255.0f, sum));
            }
        }
        
        __global__ void edge_detection_kernel(unsigned char *input, unsigned char *output,
                                            int width, int height, int channels) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= width || y >= height) return;
            
            // Sobel X kernel
            float sobel_x[9] = {
                -1, 0, 1,
                -2, 0, 2,
                -1, 0, 1
            };
            
            // Sobel Y kernel
            float sobel_y[9] = {
                -1, -2, -1,
                0, 0, 0,
                1, 2, 1
            };
            
            for (int c = 0; c < channels; c++) {
                float grad_x = 0.0f;
                float grad_y = 0.0f;
                
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        nx = max(0, min(width - 1, nx));
                        ny = max(0, min(height - 1, ny));
                        
                        int idx = (ny * width + nx) * channels + c;
                        int kidx = (ky + 1) * 3 + (kx + 1);
                        
                        grad_x += input[idx] * sobel_x[kidx];
                        grad_y += input[idx] * sobel_y[kidx];
                    }
                }
                
                float magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y);
                int out_idx = (y * width + x) * channels + c;
                output[out_idx] = (unsigned char)min(255.0f, magnitude);
            }
        }
        """)
        
        # Get kernel functions
        self.gaussian_blur = self.mod.get_function("gaussian_blur_kernel")
        self.sharpen = self.mod.get_function("sharpen_kernel")
        self.edge_detection = self.mod.get_function("edge_detection_kernel")
    
    def process_image_gpu(self, image: np.ndarray, filter_type: str) -> Tuple[np.ndarray, float]:
        """
        Process image on GPU with specified filter
        
        Args:
            image: Input image as numpy array
            filter_type: Type of filter ('blur', 'sharpen', 'edge')
            
        Returns:
            Tuple of (processed_image, processing_time)
        """
        height, width, channels = image.shape
        
        # Allocate GPU memory
        input_gpu = cuda.mem_alloc(image.nbytes)
        output_gpu = cuda.mem_alloc(image.nbytes)
        
        # Copy image to GPU
        cuda.memcpy_htod(input_gpu, image)
        
        # Configure CUDA grid and block dimensions
        block_size = (16, 16, 1)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1],
            1
        )
        
        # Start timing
        start_time = time.time()
        
        # Apply selected filter
        if filter_type == 'blur':
            self.gaussian_blur(
                input_gpu, output_gpu,
                np.int32(width), np.int32(height), np.int32(channels),
                block=block_size, grid=grid_size
            )
        elif filter_type == 'sharpen':
            self.sharpen(
                input_gpu, output_gpu,
                np.int32(width), np.int32(height), np.int32(channels),
                block=block_size, grid=grid_size
            )
        elif filter_type == 'edge':
            self.edge_detection(
                input_gpu, output_gpu,
                np.int32(width), np.int32(height), np.int32(channels),
                block=block_size, grid=grid_size
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Wait for GPU to finish
        cuda.Context.synchronize()
        end_time = time.time()
        
        # Copy result back to host
        result = np.empty_like(image)
        cuda.memcpy_dtoh(result, output_gpu)
        
        # Clean up GPU memory
        input_gpu.free()
        output_gpu.free()
        
        return result, end_time - start_time
    
    def process_image_cpu(self, image: np.ndarray, filter_type: str) -> Tuple[np.ndarray, float]:
        """
        Process image on CPU for comparison
        
        Args:
            image: Input image as numpy array
            filter_type: Type of filter ('blur', 'sharpen', 'edge')
            
        Returns:
            Tuple of (processed_image, processing_time)
        """
        start_time = time.time()
        
        if filter_type == 'blur':
            result = cv2.GaussianBlur(image, (5, 5), 0)
        elif filter_type == 'sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            result = cv2.filter2D(image, -1, kernel)
        elif filter_type == 'edge':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            edges = np.uint8(np.absolute(edges))
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        end_time = time.time()
        return result, end_time - start_time


def main():
    """Main function to handle command line arguments and process images"""
    parser = argparse.ArgumentParser(description='GPU-Accelerated Image Processing')
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('output', help='Output image file path')
    parser.add_argument('--filter', choices=['blur', 'sharpen', 'edge'], 
                       default='blur', help='Filter type to apply')
    parser.add_argument('--compare-cpu', action='store_true', 
                       help='Compare GPU vs CPU performance')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Load image
    try:
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not load image '{args.input}'")
            return 1
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1
    
    if args.verbose:
        print(f"Loaded image: {args.input}")
        print(f"Image dimensions: {image.shape}")
        print(f"Filter type: {args.filter}")
    
    # Initialize GPU processor
    try:
        processor = GPUImageProcessor()
        if args.verbose:
            print("GPU processor initialized successfully")
    except Exception as e:
        print(f"Error initializing GPU processor: {e}")
        return 1
    
    # Process image on GPU
    try:
        gpu_result, gpu_time = processor.process_image_gpu(image, args.filter)
        if args.verbose:
            print(f"GPU processing time: {gpu_time:.4f} seconds")
    except Exception as e:
        print(f"Error during GPU processing: {e}")
        return 1
    
    # Optionally compare with CPU
    if args.compare_cpu:
        try:
            cpu_result, cpu_time = processor.process_image_cpu(image, args.filter)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            print(f"\nPerformance Comparison:")
            print(f"CPU processing time: {cpu_time:.4f} seconds")
            print(f"GPU processing time: {gpu_time:.4f} seconds")
            print(f"Speedup: {speedup:.2f}x")
            
            # Save CPU result for comparison
            cpu_output = args.output.replace('.', '_cpu.')
            cv2.imwrite(cpu_output, cpu_result)
            if args.verbose:
                print(f"CPU result saved to: {cpu_output}")
        
        except Exception as e:
            print(f"Error during CPU processing: {e}")
    
    # Save GPU result
    try:
        cv2.imwrite(args.output, gpu_result)
        print(f"GPU processed image saved to: {args.output}")
    except Exception as e:
        print(f"Error saving output image: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
