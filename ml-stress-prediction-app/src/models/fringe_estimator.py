import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

class FringeOrderEstimator:
    """Neural network for fringe order estimation."""
    
    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        """Build model for fringe order estimation."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            
            layers.Conv2D(1, (1, 1), activation='linear', padding='same')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        
        return history
    
    def predict(self, X):
        """Predict fringe order map."""
        return self.model.predict(X)
    
    def save(self, model_path):
        """Save model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
    
    def load(self, model_path):
        """Load model."""
        self.model = keras.models.load_model(model_path)