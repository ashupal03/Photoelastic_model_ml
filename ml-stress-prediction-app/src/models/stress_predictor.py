import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

class StressPredictionCNN:
    """CNN model for stress prediction from photoelastic images."""
    
    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self):
        """Build CNN model for stress prediction."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        history = self.model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        )
        
        return history
    
    def predict(self, X):
        """Predict stress from images."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)
    
    def save(self, model_path):
        """Save model and scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
    
    def load(self, model_path):
        """Load model and scaler."""
        self.model = keras.models.load_model(model_path)
        scaler_path = model_path.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)