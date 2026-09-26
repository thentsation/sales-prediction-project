import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from models.sales_model import DEFAULT_MODEL_PATH, SalesPredictor

app = FastAPI(
    title="Sales Prediction API",
    description="Predicts sale amount from transaction features using a RandomForest "
    "model, trained on datetime/cash_type/card/coffee_name (see README for encoding).",
    version="1.0.0",
)

_predictor: SalesPredictor | None = None


def get_predictor() -> SalesPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SalesPredictor.load(DEFAULT_MODEL_PATH)
    return _predictor


class SalesRequest(BaseModel):
    features: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[datetime_epoch_seconds, cash_type_encoded, card_encoded, "
        "coffee_name_encoded]",
    )


class SalesResponse(BaseModel):
    predicted_sales: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=SalesResponse)
def predict_sales(request: SalesRequest) -> SalesResponse:
    predictor = get_predictor()
    prediction = predictor.predict(np.array([request.features]))[0]
    return SalesResponse(predicted_sales=float(prediction))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
