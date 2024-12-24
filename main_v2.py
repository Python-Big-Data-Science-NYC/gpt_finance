from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from joblib import dump, load
import numpy as np
import os

# Create and save a sample logistic regression model if not exists
MODEL_PATH = 'logistic_model.joblib'
if not os.path.exists(MODEL_PATH):
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    initial_model = LogisticRegression()
    initial_model.fit(X, y)
    dump(initial_model, MODEL_PATH)

# Load the saved logistic regression model
model = load(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# Retrain the model
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        data = request.json
        X_new = np.array(data['X'])
        y_new = np.array(data['y'])
        model.fit(X_new, y_new)
        dump(model, MODEL_PATH)
        return jsonify({"message": "Model retrained and saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)})

# Update model features
@app.route('/update_features', methods=['POST'])
def update_features():
    try:
        data = request.json
        X = np.array(data['X'])
        feature_indices = data['feature_indices']
        X_modified = X[:, feature_indices]
        return jsonify({"message": "Features updated successfully.", "X_modified": X_modified.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

# Update hyperparameters
@app.route('/update_hyperparams', methods=['POST'])
def update_hyperparams():
    try:
        params = request.json
        model.set_params(**params)
        dump(model, MODEL_PATH)
        return jsonify({"message": "Hyperparameters updated successfully."})
    except Exception as e:
        return jsonify({"error": str(e)})

# Predict using the model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X = np.array(data['X'])
        predictions = model.predict(X)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
