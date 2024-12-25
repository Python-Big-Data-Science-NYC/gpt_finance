from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

# Example dataset
data = pd.DataFrame({
    "income": [50000, 30000, 45000, 60000, 70000],
    "debt_to_income_ratio": [0.2, 0.5, 0.4, 0.1, 0.25],
    "credit_score": [700, 600, 650, 750, 720],
    "loan_amount": [20000, 15000, 25000, 10000, 20000],
    "default": [0, 1, 0, 0, 1]  # 0 = Non-default, 1 = Default
})

# FastAPI App
app = FastAPI()

# In-memory Logistic Regression model
model = LogisticRegression()
X = data.drop("default", axis=1)
y = data["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Define the input schema for retraining
class RetrainRequest(BaseModel):
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 100
    test_size: float = 0.2

@app.post("/retrain/")
def retrain(request: RetrainRequest):
    global model, X_train, X_test, y_train, y_test
    try:
        # Retrain model with new parameters
        model = LogisticRegression(
            penalty=request.penalty,
            solver=request.solver,
            max_iter=request.max_iter
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=42)
        model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        return {
            "message": "Model retrained successfully",
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/")
def predict(features: dict):
    try:
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0, 1]
        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
