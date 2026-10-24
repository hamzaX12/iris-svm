from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris SVM API", version="1.0.0")

# Load model at startup
model = joblib.load("iris_svm.joblib")

CLASSES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


@app.get("/")
def root():
    return {"message": "Iris SVM API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    prediction = int(model.predict(X)[0])
    return PredictionResponse(prediction=prediction, class_name=CLASSES[prediction])
