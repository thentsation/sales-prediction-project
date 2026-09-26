import mlflow
import mlflow.sklearn

from logger import setup_logging
from models.sales_model import SalesPredictor
from utils.data_loader import load_data


def main() -> tuple[float, float]:
    logger = setup_logging()
    logger.info("Starting training")

    mlflow.set_experiment("sales-prediction")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data(filepath="data/sales_data.csv")

        model = SalesPredictor(logger=logger)
        model.train(X_train, y_train)

        mae, mse = model.evaluate(X_test, y_test)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        model.save_model()

    logger.info("Model trained and saved. MAE: %.4f, MSE: %.4f", mae, mse)
    return mae, mse


if __name__ == "__main__":
    main()
