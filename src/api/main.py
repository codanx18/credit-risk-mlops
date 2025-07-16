from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import time

from .schemas import CreditDataInput, PredictionOutput

app = FastAPI()

# Load models
model = joblib.load('models/model.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')

@app.post("/predict", response_model=PredictionOutput)
def predict(data: CreditDataInput):
    "Prediccion basada en los datos de entrada."
    input_df = pd.DataFrame([data.dict()])
    processed_input = preprocessor.transform(input_df)
    
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)[:, 1]
    
    return PredictionOutput(prediction=prediction.tolist(), probability=probability.tolist())

# # Endpoint de ejemplo para probar
# @app.get("/")
# def read_root():
#     return {"message": "API de Riesgo Crediticio funcionando"}    