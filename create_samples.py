#!/usr/bin/env python3
"""
Sample Image Generator for GPU Image Processing Capstone
Creates realistic test images when actual samples are not available
"""

import cv2
import numpy as np
import os
from typing import Tuple

def create_landscape_image(size: Tuple[int, int] = (720, 1280)) -> np.ndarray:
    """Create a synthetic landscape image with gradients and patterns"""
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky gradient (blue to light blue)
    sky_height = height // 3
    for y in range(sky_height):
        intensity = int(120 + (180 - 120) * (1 - y / sky_height))
        image[y, :] = [intensity, intensity - 30, intensity - 60]
    
    # Mountains (dark gray/brown)
    mountain_start = sky_height
    mountain_height = height // 4
    for y in range(mountain_start, mountain_start + mountain_height):
        noise = np.random.randint(-20, 20, width)
        base_color = 80 + noise
        image[y, :] = [base_color, base_color - 10, base_color - 20]
    
    # Ground (green/brown)
    ground_start = mountain_start + mountain_height
    for y in range(ground_start, height):
        noise = np.random.randint(-15, 15, width)
        green_intensity = 100 + noise
        image[y, :] = [green_intensity - 40, green_intensity, green_intensity - 60]
    
    # Add some texture
    texture = np.random.randint(-10, 10, (height, width, 3))
    image = np.clip(image.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    
    return image

def create_portrait_image(size: Tuple[int, int] = (960, 720)) -> np.ndarray:
    """Create a synthetic portrait-like image"""
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(height):
        for x in range(width):
            intensity = int(150 + 50 * np.sin(x / width * np.pi) * np.cos(y / height * np.pi))
            image[y, x] = [intensity, intensity - 20, intensity - 40]
    
    # Add circular "face" region with different color
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 <= radius ** 2
    
    # Face region (warmer tones)
    face_color = np.random.randint(180, 220, 3)
    image[mask] = face_color
    
    # Add noise for realism
    noise = np.random.randint(-15, 15, (height, width, 3))
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def create_architecture_image(size: Tuple[int, int] = (800, 1200)) -> np.ndarray:
    """Create a synthetic architectural image with geometric patterns"""
    height, width = size
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    
    # Create vertical "building" stripes
    stripe_width = width // 8
    for i in range(0, width, stripe_width * 2):
        end_x = min(i + stripe_width, width)
        color_intensity = np.random.randint(80, 180)
        image[:, i:end_x] = [color_intensity, color_intensity - 20, color_intensity - 30]
    
    # Add horizontal "floor" lines
    floor_spacing = height // 12
    for y in range(floor_spacing, height, floor_spacing):
        if y < height:
            image[y:y+3, :] = [60, 60, 60]  # Dark horizontal lines
    
    # Add window-like rectangles
    for building in range(0, width, stripe_width * 2):
        for floor in range(floor_spacing, height - floor_spacing, floor_spacing):
            window_x = building + stripe_width // 4
            window_y = floor + floor_spacing // 4
            window_w = stripe_width // 2
            window_h = floor_spacing // 2
            
            if window_x + window_w < width and window_y + window_h < height:
                # Random window color (some lit, some dark)
                if np.random.random() > 0.3:
                    window_color = [200, 220, 180]  # Lit window
                else:
                    window_color = [40, 50, 60]     # Dark window
                
                image[window_y:window_y+window_h, window_x:window_x+window_w] = window_color
    
    # Add texture noise
    noise = np.random.randint(-8, 8, (height, width, 3))
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def create_nature_image(size: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
    """Create a synthetic nature image with organic patterns"""
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create organic-looking background with multiple frequency noise
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Multi-frequency pattern for organic look
    pattern1 = np.sin(X) * np.cos(Y)
    pattern2 = np.sin(2 * X + np.pi/4) * np.cos(2 * Y + np.pi/4)
    pattern3 = np.sin(0.5 * X) * np.cos(0.5 * Y)
    
    combined_pattern = pattern1 + 0.5 * pattern2 + 0.3 * pattern3
    
    # Convert to color (green nature theme)
    base_green = 120
    green_channel = base_green + (combined_pattern * 50).astype(np.int32)
    red_channel = (base_green - 40) + (combined_pattern * 30).astype(np.int32)
    blue_channel = (base_green - 60) + (combined_pattern * 20).astype(np.int32)
    
    image[:, :, 0] = np.clip(blue_channel, 0, 255)
    image[:, :, 1] = np.clip(green_channel, 0, 255)
    image[:, :, 2] = np.clip(red_channel, 0, 255)
    
    # Add random "leaf" or "rock" spots
    num_spots = 50
    for _ in range(num_spots):
        spot_x = np.random.randint(0, width - 20)
        spot_y = np.random.randint(0, height - 20)
        spot_size = np.random.randint(5, 25)
        
        spot_color = np.random.choice([
            [80, 140, 60],   # Dark green
            [100, 80, 70],   # Brown
            [60, 100, 80],   # Medium green
        ])
        
        # Create circular spot
        y_coords, x_coords = np.ogrid[:height, :width]
        mask = (x_coords - spot_x) ** 2 + (y_coords - spot_y) ** 2 <= spot_size ** 2
        image[mask] = spot_color
    
    return image

def create_abstract_image(size: Tuple[int, int] = (600, 800)) -> np.ndarray:
    """Create a synthetic abstract image with geometric shapes"""
    height, width = size
    image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add geometric shapes
    shapes = ['circle', 'rectangle', 'triangle']
    colors = [
        [255, 100, 100],  # Red
        [100, 255, 100],  # Green
        [100, 100, 255],  # Blue
        [255, 255, 100],  # Yellow
        [255, 100, 255],  # Magenta
        [100, 255, 255],  # Cyan
    ]
    
    num_shapes = 15
    for _ in range(num_shapes):
        shape = np.random.choice(shapes)
        color = np.random.choice(colors)
        
        if shape == 'circle':
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(20, 80)
            
            y_coords, x_coords = np.ogrid[:height, :width]
            mask = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 <= radius ** 2
            image[mask] = color
            
        elif shape == 'rectangle':
            x1 = np.random.randint(0, width - 50)
            y1 = np.random.randint(0, height - 50)
            w = np.random.randint(30, 100)
            h = np.random.randint(30, 100)
            x2 = min(x1 + w, width)
            y2 = min(y1 + h, height)
            
            image[y1:y2, x1:x2] = color
    
    # Add some gradient overlay
    for y in range(height):
        alpha = 0.3 * np.sin(y / height * np.pi)
        image[y, :] = image[y, :] * (1 - alpha) + np.array([128, 128, 128]) * alpha
    
    return image.astype(np.uint8)

def main():
    """Generate all sample images"""
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    generators = {
        'landscape.jpg': create_landscape_image,
        'portrait.jpg': create_portrait_image,
        'architecture.jpg': create_architecture_image,
        'nature.jpg': create_nature_image,
        'abstract.jpg': create_abstract_image
    }
    
    print("Generating sample images...")
    
    for filename, generator in generators.items():
        filepath = os.path.join(samples_dir, filename)
        
        if os.path.exists(filepath):
            print(f"  {filename} already exists, skipping...")
            continue
        
        print(f"  Creating {filename}...")
        image = generator()
        
        # Add some final processing for realism
        # Slight blur to make it look more natural
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Save image
        cv2.imwrite(filepath, image)
        print(f"  ✓ Saved {filepath} ({image.shape[1]}x{image.shape[0]})")
    
    print(f"\nSample generation complete!")
    print(f"Generated images are available in the '{samples_dir}' directory.")

if __name__ == "__main__":
    main()
