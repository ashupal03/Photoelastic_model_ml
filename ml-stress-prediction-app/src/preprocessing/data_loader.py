import numpy as np
import cv2
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from .image_preprocessing import ImagePreprocessor

class DataLoader:
    """Load and manage photoelastic image data."""
    
    def __init__(self, preprocessor=None):
        """
        Initialize DataLoader.
        
        Args:
            preprocessor: ImagePreprocessor instance
        """
        self.preprocessor = preprocessor or ImagePreprocessor()
    
    def load_single_image(self, image_path):
        """Load and preprocess single image."""
        try:
            result = self.preprocessor.preprocess(image_path)
            return result['normalized'].reshape(256, 256, 1)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def load_images_from_directory(self, directory, max_images=None):
        """
        Load all images from directory.
        
        Args:
            directory: Path to image directory
            max_images: Maximum number of images to load
            
        Returns:
            Tuple of (images, filenames)
        """
        images = []
        filenames = []
        
        image_files = sorted(Path(directory).glob('*.jpg')) + \
                     sorted(Path(directory).glob('*.png')) + \
                     sorted(Path(directory).glob('*.tif')) + \
                     sorted(Path(directory).glob('*.bmp'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        for image_path in image_files:
            img = self.load_single_image(image_path)
            if img is not None:
                images.append(img)
                filenames.append(image_path.name)
        
        return np.array(images), filenames
    
    def load_with_labels(self, image_dir, label_file=None):
        """
        Load images with corresponding labels.
        
        Args:
            image_dir: Directory containing images
            label_file: CSV file with image names and stress values
            
        Returns:
            Tuple of (images, labels)
        """
        images, filenames = self.load_images_from_directory(image_dir)
        
        if label_file and os.path.exists(label_file):
            import pandas as pd
            labels_df = pd.read_csv(label_file)
            labels = []
            
            for filename in filenames:
                label = labels_df[labels_df['filename'] == filename]['stress'].values
                if len(label) > 0:
                    labels.append(float(label[0]))
            
            return images, np.array(labels)
        else:
            # Generate synthetic labels for demonstration
            labels = np.random.uniform(1e6, 1e8, len(images))
            return images, labels
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input data
            y: Labels
            test_size: Test set ratio
            val_size: Validation set ratio
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_batches(self, X, y, batch_size=32, shuffle=True):
        """
        Create batches for training.
        
        Args:
            X: Input data
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            Generator yielding (batch_X, batch_y)
        """
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield X[batch_indices], y[batch_indices]

class ImageDataGenerator:
    """Generate batches of augmented images."""
    
    def __init__(self, augmentation_config=None):
        """Initialize image data generator."""
        self.augmentation_config = augmentation_config or {}
    
    def flow_from_directory(self, directory, batch_size=32, target_size=(256, 256)):
        """Flow images from directory."""
        preprocessor = ImagePreprocessor(target_size=target_size)
        images, _ = DataLoader(preprocessor).load_images_from_directory(directory)
        
        for i in range(0, len(images), batch_size):
            yield images[i:i + batch_size]