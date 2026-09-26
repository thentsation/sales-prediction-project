import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLUMNS = ["cash_type", "card", "coffee_name"]
TARGET_COLUMN = "money"

SalesSplit = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def load_data(
    filepath: str = "data/sales_data.csv", test_size: float = 0.2
) -> SalesSplit:
    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"{filepath} contains no rows")

    df = df.dropna()
    if df.empty:
        raise ValueError(f"{filepath} became empty after dropping rows with NaN values")

    missing_categorical = [
        c for c in [*CATEGORICAL_COLUMNS, TARGET_COLUMN] if c not in df.columns
    ]
    if missing_categorical:
        raise ValueError(f"Missing expected column(s): {missing_categorical}")

    df["datetime"] = pd.to_datetime(df["datetime"]).astype("int64") / 10**9

    for column in CATEGORICAL_COLUMNS:
        df[column] = LabelEncoder().fit_transform(df[column])

    X = df.drop(columns=["date", TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(X, y, test_size=test_size, random_state=42)
