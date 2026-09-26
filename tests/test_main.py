import main as main_module


def test_main_trains_and_returns_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        main_module.SalesPredictor,
        "save_model",
        lambda self, path=None: None,
    )

    mae, mse = main_module.main()

    assert mae >= 0
    assert mse >= 0
