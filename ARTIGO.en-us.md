[🇧🇷 Português](ARTIGO.md) | 🇺🇸 English

# The API that never scaled its features before predicting

This was the most serious bug I found across this whole productization pass, and the quietest: the API answered 200, returned a number, and that number was systematically wrong.

## What I had

A sales-prediction pipeline — `RandomForestRegressor` with `StandardScaler`, training via `train.py`, MLflow tracking, FastAPI serving — spread across a duplicated `api/`, `src/models/`, `src/utils/` structure at the root. `data_loader.py` swallowed every exception and silently returned `None` in several places, which didn't actually protect anything in practice: `X_train, X_test, y_train, y_test = load_data(...)` can't unpack `None` into four variables, so the "error handling" only postponed the crash into a context-free `TypeError` one line later.

## The real bug: `save_model` saved the model, not the predictor

```python
# original
def save_model(self, path="models/sales_predictor.pkl"):
    joblib.dump(self.model, path)
```

`self.model` is just the inner `RandomForestRegressor`. The `StandardScaler` — fitted during `train()`, used inside `predict()` to scale input before predicting — was never saved. The API loaded that pickle and called `.predict()` **directly on the RandomForest**, skipping the scaler entirely:

```python
# original api/main.py
model = joblib.load("models/sales_predictor.pkl")
...
prediction = model.predict(np.array([request.features]))[0]
```

In other words: every prediction in production used unscaled features, while the model was trained on scaled ones. No error, no exception raised — just a systematically off number, because `RandomForestRegressor` isn't as scale-sensitive as a linear model, but still behaves differently on input magnitudes it wasn't fit on. This shows up in no log, no test, no health check. It only shows up when someone compares the prediction against ground truth.

The fix was to change what gets serialized: instead of `self.model`, `save_model` now serializes **the whole `SalesPredictor`** (which holds `model` and `scaler` together), and `load()` returns that same instance:

```python
def save_model(self, path: str = DEFAULT_MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(self, path)

@staticmethod
def load(path: str = DEFAULT_MODEL_PATH) -> "SalesPredictor":
    predictor = joblib.load(path)
    if not isinstance(predictor, SalesPredictor):
        raise TypeError(f"{path} does not contain a SalesPredictor")
    return predictor
```

The test that would have caught this — and that I wrote as soon as I understood the problem — trains, saves, reloads, and compares the reloaded object's prediction against the original:

```python
def test_save_and_load_roundtrip_preserves_scaling(tmp_path) -> None:
    predictor = SalesPredictor()
    predictor.train(X, y)
    expected = predictor.predict(X)

    predictor.save_model(str(tmp_path / "model.pkl"))
    loaded = SalesPredictor.load(str(tmp_path / "model.pkl"))

    np.testing.assert_array_almost_equal(loaded.predict(X), expected)
```

## `logging.info` that never logged anything

`train.py` called `logging.info("Starting training!")` at module scope, without ever configuring logging (`logging.basicConfig`). Without configuration, Python's root logger uses a handler that drops everything below `WARNING` — so those `info` calls never showed up anywhere. I replaced this with a real `setup_logging()` (the same pattern used across the other projects in the org) and passed the logger explicitly into `SalesPredictor`, swapping the scattered `print()` calls for real logging.

## Explicit errors instead of silent `None`

I rewrote `data_loader.load_data` to raise a clear `ValueError` in every case that used to return `None` — empty file, missing column, everything dropped after `dropna`. A `ValueError("Missing expected column(s): [...]")` at the right spot is infinitely easier to debug than a generic `TypeError` three frames downstream.

## What I didn't change

The API contract still expects already-encoded features (`[datetime_epoch, cash_type_encoded, card_encoded, coffee_name_encoded]`) instead of accepting raw categorical strings. The `LabelEncoder`s used during training aren't persisted or versioned — whoever calls the API needs to know the encoding upfront. I didn't redesign this contract: it's a real limitation of the original project, now documented in the README, but changing how the API receives input is a product decision that wasn't mine to make unilaterally on this pass — the bug worth fixing without asking was the scaler never being applied, not the API's design.
