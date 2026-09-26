import numpy as np
import pandas as pd

from models.sales_model import SalesPredictor

X = pd.DataFrame(
    {
        "datetime": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cash_type": [0, 1, 0, 1, 0, 1],
        "card": [0, 1, 2, 0, 1, 2],
        "coffee_name": [0, 1, 2, 3, 0, 1],
    }
)
y = pd.Series([10.0, 20.0, 15.0, 25.0, 12.0, 22.0])


def test_train_predict_evaluate() -> None:
    predictor = SalesPredictor()
    predictor.train(X, y)

    predictions = predictor.predict(X)
    assert predictions.shape == (6,)

    mae, mse = predictor.evaluate(X, y)
    assert mae >= 0
    assert mse >= 0


def test_save_and_load_roundtrip_preserves_scaling(tmp_path) -> None:
    predictor = SalesPredictor()
    predictor.train(X, y)
    expected = predictor.predict(X)

    path = str(tmp_path / "model.pkl")
    predictor.save_model(path)

    loaded = SalesPredictor.load(path)
    np.testing.assert_array_almost_equal(loaded.predict(X), expected)
