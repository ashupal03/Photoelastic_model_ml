import os
import cv2
import numpy as np
from ..models.stress_predictor import StressPredictionCNN
from ..preprocessing.image_preprocessing import ImagePreprocessor
from ..utils.helpers import normalize_image

class StressPredictor:
    """Predict stress using trained ML model."""
    
    def __init__(self, model_path=None):
        self.model = StressPredictionCNN()
        if model_path is not None:
            self.model.load(model_path)
        else:
            # Default model path
            self.model.load(os.path.join("models", "stress_prediction_model.h5"))
        self.preprocessor = ImagePreprocessor()
    
    def predict_from_image(self, image_path):
        """Predict stress from image file."""
        result = self.preprocessor.preprocess(image_path)
        img_preprocessed = result['preprocessed']
        img_normalized = normalize_image(img_preprocessed)
        img_batch = img_normalized.reshape(1, 256, 256, 1)
        stress = self.model.predict(img_batch)
        return {
            'image': image_path,
            'predicted_stress': float(stress[0][0]),
            'preprocessed_image': img_preprocessed
        }
    
    def predict_batch(self, image_dir):
        """Predict stress for all images in directory."""
        results = []
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.png', '.tif', '.bmp')):
                image_path = os.path.join(image_dir, filename)
                try:
                    result = self.predict_from_image(image_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error predicting {filename}: {e}")
        return results

if __name__ == "__main__":
    predictor = StressPredictor()
    result = predictor.predict_from_image("test_image.png")
    print(result)