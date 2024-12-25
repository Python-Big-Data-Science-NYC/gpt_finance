from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics import classification_report

MODEL_PATH = "default_prediction_model.pkl"

app = FastAPI()
model = joblib.load(MODEL_PATH)  # Load the saved logistic regression model

class UserInput(BaseModel):
    message: str

@app.post("/chat")
def chat_with_model(user_input: UserInput):
    message = user_input.message.lower()
    
    if "rerun the model" in message:
        # Retrain the model with new dummy data
        X_new = np.random.random((100, 6))
        y_new = np.random.choice([0, 1], size=100)
        model.fit(X_new, y_new)
        joblib.dump(model, MODEL_PATH)
        return {"response": "The model has been retrained with new data."}
    
    elif "model performance" in message:
        # Evaluate model on dummy test data
        X_test = np.random.random((50, 6))
        y_test = np.random.choice([0, 1], size=50)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {"response": "Model performance evaluated.", "performance": report}
    
    elif "predict default" in message:
        # Predict with a sample input
        example_input = np.array([[45000, 2.1, 65, 620, 250000, 6]])  # Replace with actual inputs
        probability = model.predict_proba(example_input)
        return {
            "response": "Prediction completed.",
            "default_probability": round(probability[0][1], 4),
            "non_default_probability": round(probability[0][0], 4),
            "prediction": "Default" if probability[0][1] > 0.5 else "No Default"
        }
    
    else:
        return {"response": "Sorry, I didn’t understand. Try 'rerun the model', 'model performance', or 'predict default'."}

