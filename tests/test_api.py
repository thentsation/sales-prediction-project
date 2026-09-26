import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main
from models.sales_model import SalesPredictor


def _trained_predictor() -> SalesPredictor:
    X = pd.DataFrame(
        {
            "datetime": [1.0, 2.0, 3.0, 4.0],
            "cash_type": [0, 1, 0, 1],
            "card": [0, 1, 2, 0],
            "coffee_name": [0, 1, 2, 3],
        }
    )
    y = pd.Series([10.0, 20.0, 15.0, 25.0])
    predictor = SalesPredictor()
    predictor.train(X, y)
    return predictor


def test_health() -> None:
    client = TestClient(api_main.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_uses_the_loaded_predictor(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "_predictor", _trained_predictor())
    client = TestClient(api_main.app)

    response = client.post("/predict", json={"features": [1.0, 0, 0, 0]})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_sales"], float)


def test_predict_rejects_wrong_feature_count() -> None:
    client = TestClient(api_main.app)
    response = client.post("/predict", json={"features": [1.0, 0, 0]})
    assert response.status_code == 422


def test_get_predictor_loads_and_caches_on_first_call(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "_predictor", None)
    loaded = _trained_predictor()
    monkeypatch.setattr(SalesPredictor, "load", staticmethod(lambda path: loaded))

    assert api_main.get_predictor() is loaded
    assert api_main.get_predictor() is loaded
