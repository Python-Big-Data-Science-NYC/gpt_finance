import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "default_prediction_model.pkl"

def initialize_model():
    """Initializes or loads the logistic regression model."""
    try:
        # Attempt to load the model if it exists
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except FileNotFoundError:
        # If file doesn't exist, create and save the model
        print("Model file not found. Creating a new model...")
        model = LogisticRegression()
        # Create dummy training data
        X_dummy = np.random.random((100, 6))  # Replace with real data in production
        y_dummy = np.random.choice([0, 1], size=100)
        model.fit(X_dummy, y_dummy)
        joblib.dump(model, MODEL_PATH)
        print(f"New model created and saved to {MODEL_PATH}.")
    return model

if __name__ == "__main__":
    initialize_model()
