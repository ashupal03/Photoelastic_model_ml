import numpy as np
import cv2
import os
from pathlib import Path

def create_directories(config):
    """Create necessary directories from config."""
    directories = [
        config['paths']['data_dir'],
        config['paths']['raw_data_dir'],
        config['paths']['processed_data_dir'],
        config['paths']['models_dir'],
        config['paths']['logs_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created successfully")

def normalize_image(img, dtype=np.float32):
    """Normalize image to [0, 1] range."""
    return img.astype(dtype) / 255.0

def denormalize_image(img):
    """Denormalize image from [0, 1] to [0, 255]."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def resize_batch(images, target_size=(256, 256)):
    """Resize batch of images."""
    resized = []
    for img in images:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        resized.append(resized_img)
    return np.array(resized)

def save_model_info(model, filepath):
    """Save model summary to file."""
    with open(filepath, 'w') as f:
        model.model.summary(print_fn=lambda x: f.write(x + '\n'))

def get_image_files(directory, extensions=['.jpg', '.png', '.tif', '.bmp']):
    """Get all image files from directory."""
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(image_files)

def calculate_statistics(data):
    """Calculate statistics for data."""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data)
    }

def log_message(message, log_file="logs/app.log"):
    """Log message to file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")