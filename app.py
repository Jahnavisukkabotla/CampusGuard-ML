
import os
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Trained model not found at {MODEL_PATH}. Train the model first."
    )

model = joblib.load(MODEL_PATH)
app = FastAPI(title="CampusGuard ML API", version="1.0.0")


class EmailRequest(BaseModel):
    subject: str = ""
    body: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(email: EmailRequest):
    text = f"{email.subject or ''} {email.body or ''}".strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Email subject or body is required",
        )

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    probability_map = {
        label: float(probabilities[index])
        for index, label in enumerate(classes)
    }

    phishing_probability = probability_map.get("phishing", 0.0)
    risk_score = round(phishing_probability * 100)

    # These are initial product thresholds, not calibrated guarantees.
    if risk_score >= 70:
        status = "PHISHING"
    elif risk_score >= 40:
        status = "SUSPICIOUS"
    else:
        status = "SAFE"

    return {
        "prediction": prediction,
        "status": status,
        "riskScore": risk_score,
        "modelProbability": round(phishing_probability, 4),
    }
