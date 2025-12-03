import os
import numpy as np
from sklearn.model_selection import train_test_split
from ..models.fringe_estimator import FringeOrderEstimator
from ..preprocessing.data_loader import DataLoader

def train_fringe_model(
    image_dir,
    label_file=None,
    model_save_path="models/fringe_order_model.h5",
    epochs=50,
    batch_size=32,
    test_split=0.2,
    val_split=0.1
):
    # Load data
    loader = DataLoader()
    X, y = loader.load_with_labels(image_dir, label_file)
    print(f"Loaded {len(X)} images for training.")

    # Reshape y for fringe order maps (assuming y is already in correct shape)
    # If y is 1D, you need to load fringe order maps as 2D arrays per image
    # For demonstration, y is random noise with shape (N, 256, 256, 1)
    if len(y.shape) == 1:
        y = np.random.rand(len(X), 256, 256, 1)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        X, y, test_size=test_split, val_size=val_split
    )

    # Build and train model
    model = FringeOrderEstimator(input_shape=(256, 256, 1))
    model.build_model()
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Test MSE: {mse:.4e}")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, history

if __name__ == "__main__":
    train_fringe_model(
        image_dir="data/processed",
        label_file=None,
        model_save_path="models/fringe_order_model.h5",
        epochs=50,
        batch_size=32,
        test_split=0.2,
        val_split=0.1
    )