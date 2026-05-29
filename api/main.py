from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

pipeline = joblib.load(
    "models/churn_pipeline.pkl"
)

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn API Running"
    }


@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame(
        [customer.model_dump()]
    )

    prediction = pipeline.predict(data)[0]

    return {
        "prediction": prediction
    }