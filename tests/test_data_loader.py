import pandas as pd
import pytest

from utils.data_loader import load_data

CSV_HEADER = "date,datetime,cash_type,card,money,coffee_name\n"


def test_load_data_returns_encoded_numeric_splits(tmp_path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        CSV_HEADER
        + "2024-03-01,2024-03-01 10:15:50.520,card,ANON-1,38.7,Latte\n"
        + "2024-03-01,2024-03-01 12:19:22.539,card,ANON-2,38.7,Hot Chocolate\n"
        + "2024-03-02,2024-03-02 09:00:00.000,cash,ANON-1,25.0,Latte\n"
        + "2024-03-02,2024-03-02 09:05:00.000,cash,ANON-2,25.0,Espresso\n"
    )

    X_train, X_test, y_train, y_test = load_data(str(csv_path), test_size=0.5)

    assert list(X_train.columns) == ["datetime", "cash_type", "card", "coffee_name"]
    assert pd.api.types.is_numeric_dtype(X_train["cash_type"])
    assert len(X_train) + len(X_test) == 4
    assert len(y_train) + len(y_test) == 4


def test_load_data_raises_on_empty_file(tmp_path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(CSV_HEADER)

    with pytest.raises(ValueError, match="no rows"):
        load_data(str(csv_path))


def test_load_data_raises_on_missing_column(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("date,datetime,money\n2024-03-01,2024-03-01 10:00:00,10.0\n")

    with pytest.raises(ValueError, match="Missing expected column"):
        load_data(str(csv_path))
