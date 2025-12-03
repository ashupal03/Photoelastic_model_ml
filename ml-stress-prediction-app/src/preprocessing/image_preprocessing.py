import cv2
import numpy as np
from skimage.filters import frangi, threshold_otsu
from skimage import morphology
from PIL import Image
from torchvision import transforms

class ImagePreprocessor:
    """Preprocess photoelastic images for analysis and ML training."""
    
    def __init__(self, target_size=(256, 256), clahe_clip=2.0, clahe_tile=(8, 8)):
        """
        Initialize ImagePreprocessor.
        
        Args:
            target_size: Target image size (height, width)
            clahe_clip: CLAHE clip limit
            clahe_tile: CLAHE tile grid size
        """
        self.target_size = target_size
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
    
    def load_image(self, image_path):
        """Load image from file."""
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return img
    
    def resize_image(self, img):
        """Resize image to target size."""
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
    
    def to_grayscale(self, img):
        """Convert image to grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def apply_clahe(self, img):
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)
        return clahe.apply(img)
    
    def apply_blur(self, img, kernel_size=(5, 5)):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(img, kernel_size, 0)
    
    def sharpen(self, img, blurred):
        """Sharpen image using unsharp masking."""
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    def normalize(self, img):
        """Normalize image to [0, 1]."""
        return img.astype(np.float32) / 255.0
    
    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with preprocessed image and intermediate stages
        """
        img = self.load_image(image_path)
        img_resized = self.resize_image(img)
        img_gray = self.to_grayscale(img_resized)
        img_blurred = self.apply_blur(img_gray)
        img_clahe = self.apply_clahe(img_blurred)
        img_sharpened = self.sharpen(img_clahe, img_blurred)
        img_normalized = self.normalize(img_sharpened)
        
        return {
            'original': img_gray,
            'blurred': img_blurred,
            'clahe': img_clahe,
            'preprocessed': img_sharpened,
            'normalized': img_normalized
        }

class FringeDetector:
    """Detect fringe patterns in photoelastic images."""
    
    def __init__(self, min_sigma=1, max_sigma=10):
        """
        Initialize FringeDetector.
        
        Args:
            min_sigma: Minimum sigma for Frangi filter
            max_sigma: Maximum sigma for Frangi filter
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
    
    def detect_frangi(self, img, min_sigma=None, max_sigma=None):
        """Apply Frangi filter for fringe detection."""
        min_s = min_sigma if min_sigma else self.min_sigma
        max_s = max_sigma if max_sigma else self.max_sigma
        
        frangi_img = frangi(img, sigmas=np.arange(min_s, max_s + 1, 1))
        frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return frangi_norm
    
    def threshold_image(self, img):
        """Apply Otsu thresholding."""
        otsu_thresh = threshold_otsu(img)
        _, binary = cv2.threshold(img, otsu_thresh, 255, cv2.THRESH_BINARY)
        return binary
    
    def skeletonize(self, img):
        """Convert binary image to skeleton."""
        skeleton = morphology.skeletonize(img > 0)
        return skeleton.astype(np.uint8) * 255
    
    def detect_fringes(self, img):
        """
        Complete fringe detection pipeline.
        
        Args:
            img: Input image
            
        Returns:
            Dictionary with fringe detection results
        """
        min_s = max(1, img.shape[0] // 300)
        max_s = max(2, img.shape[0] // 100)
        
        frangi_img = self.detect_frangi(img, min_s, max_s)
        binary = self.threshold_image(frangi_img)
        skeleton = self.skeletonize(binary)
        
        return {
            'frangi': frangi_img,
            'binary': binary,
            'skeleton': skeleton
        }

class DataAugmenter:
    """Augment photoelastic images for better training."""
    
    @staticmethod
    def rotate(img, angle):
        """Rotate image."""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))
    
    @staticmethod
    def flip(img, direction='horizontal'):
        """Flip image."""
        if direction == 'horizontal':
            return cv2.flip(img, 1)
        elif direction == 'vertical':
            return cv2.flip(img, 0)
        return img
    
    @staticmethod
    def add_noise(img, noise_amount=0.01):
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, noise_amount, img.shape)
        return np.clip(img + noise, 0, 1)
    
    @staticmethod
    def adjust_brightness(img, factor=0.1):
        """Adjust image brightness."""
        return np.clip(img * (1 + factor), 0, 1)
    
    @staticmethod
    def adjust_contrast(img, factor=0.1):
        """Adjust image contrast."""
        mean = np.mean(img)
        return np.clip((img - mean) * (1 + factor) + mean, 0, 1)
    
    @classmethod
    def augment_batch(cls, images, augmentation_factor=2):
        """Augment batch of images."""
        augmented = list(images)
        
        for img in images:
            augmented.append(cls.rotate(img, np.random.uniform(-10, 10)))
            augmented.append(cls.flip(img))
            augmented.append(cls.add_noise(img))
        
        return np.array(augmented)

def resize_image(image, target_size):
    return image.resize(target_size, Image.ANTIALIAS)

def normalize_image(image):
    return (np.array(image) / 255.0).astype(np.float32)

def augment_image(image):
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    return augmentation(image)

def preprocess_image(image, target_size):
    image = resize_image(image, target_size)
    image = normalize_image(image)
    return augment_image(image)