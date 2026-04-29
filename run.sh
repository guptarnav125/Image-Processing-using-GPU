#!/bin/bash

# GPU-Accelerated Image Processing - Execution Script
# Author: GPU Programming Specialization Student

echo "GPU-Accelerated Image Processing Capstone Project"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required directories exist
if [ ! -d "samples" ]; then
    echo "Creating samples directory..."
    mkdir -p samples
fi

if [ ! -d "outputs" ]; then
    echo "Creating outputs directory..."
    mkdir -p outputs
fi

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "Warning: $1 not found"
        return 1
    fi
    return 0
}

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Warning: Some dependencies might not have installed correctly"
    fi
fi

echo ""
echo "Testing CUDA availability..."
python3 -c "
try:
    import pycuda.autoinit
    print('✓ CUDA is available and working')
except Exception as e:
    print('✗ CUDA error:', str(e))
    print('Please ensure NVIDIA drivers and CUDA toolkit are installed')
"

echo ""
echo "Running sample image processing tasks..."
echo "========================================"

# Define sample processing tasks
declare -a samples=("landscape.jpg" "portrait.jpg" "architecture.jpg" "nature.jpg" "abstract.jpg")
declare -a filters=("blur" "sharpen" "edge")

# Check if we have sample images (in practice, these would be provided)
sample_count=0
for sample in "${samples[@]}"; do
    if check_file "samples/$sample"; then
        ((sample_count++))
    fi
done

if [ $sample_count -eq 0 ]; then
    echo "No sample images found in samples/ directory."
    echo "Please add sample images or specify your own input files."
    echo ""
    echo "Usage examples:"
    echo "  python3 main.py input.jpg output.jpg --filter blur"
    echo "  python3 main.py input.jpg output.jpg --filter sharpen --compare-cpu"
    echo "  python3 main.py input.jpg output.jpg --filter edge --verbose"
    echo ""
    echo "Available filters: blur, sharpen, edge"
    exit 0
fi

echo "Found $sample_count sample image(s)."
echo ""

# Process each available sample with different filters
processed=0
for sample in "${samples[@]}"; do
    if check_file "samples/$sample"; then
        echo "Processing $sample..."
        
        # Extract filename without extension
        filename=$(basename "$sample" .jpg)
        
        for filter in "${filters[@]}"; do
            output_file="outputs/${filename}_${filter}.jpg"
            
            echo "  Applying $filter filter..."
            
            # Run the processing with performance comparison
            python3 main.py "samples/$sample" "$output_file" \
                --filter "$filter" --compare-cpu --verbose
            
            if [ $? -eq 0 ]; then
                echo "    ✓ Success: $output_file created"
                ((processed++))
            else
                echo "    ✗ Failed to process with $filter filter"
            fi
        done
        echo ""
    fi
done

echo "Processing complete!"
echo "Processed $processed image-filter combinations"
echo ""

# Display results summary
if [ $processed -gt 0 ]; then
    echo "Output files created:"
    echo "===================="
    ls -la outputs/ 2>/dev/null || echo "No output directory found"
    
    echo ""
    echo "Performance Summary:"
    echo "==================="
    echo "Check the console output above for detailed timing comparisons"
    echo "between CPU and GPU processing for each filter type."
fi

echo ""
echo "For custom processing, use:"
echo "python3 main.py <input_image> <output_image> --filter <blur|sharpen|edge> [--compare-cpu] [--verbose]"
echo ""
echo "Project completed successfully!"
