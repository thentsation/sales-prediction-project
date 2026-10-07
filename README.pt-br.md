# Sales Prediction API

[![Python CI](https://github.com/thentsation/sales-prediction-project/actions/workflows/pipeline_python.yaml/badge.svg)](https://github.com/thentsation/sales-prediction-project/actions/workflows/pipeline_python.yaml)
[![Docker CI/CD](https://github.com/thentsation/sales-prediction-project/actions/workflows/pipeline_docker.yaml/badge.svg)](https://github.com/thentsation/sales-prediction-project/actions/workflows/pipeline_docker.yaml)

> Read in [English](README.md).

Prediz o valor de venda de uma transação de cafeteria com um `RandomForestRegressor`, treinado num dataset real de ponto de venda (2176 transações), rastreado com MLflow e servido com FastAPI.

Um artigo detalhado sobre a produtização deste projeto — incluindo um scaler que nunca era aplicado de verdade na inferência — está disponível em [ARTIGO.md](ARTIGO.md) (pt-br) / [ARTIGO.en-us.md](ARTIGO.en-us.md) (en-us).

## Estrutura do projeto

```text
src/
├── models/
│   ├── base_model.py     # ModelInterface
│   └── sales_model.py     # SalesPredictor: RandomForestRegressor + StandardScaler
├── utils/data_loader.py    # CSV -> split treino/teste codificado
├── api/main.py              # FastAPI: POST /predict, GET /health
└── main.py                   # carrega -> treina -> loga (MLflow) -> salva
```

## Como rodar

```bash
make install    # cria o .venv e instala as deps
make train      # treina o modelo, loga no MLflow, salva models/sales_predictor.pkl
make serve      # uvicorn --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- UI do MLflow: `.venv/bin/mlflow ui` (lê `./mlruns`, gitignorado, gerado por `make train`)

### Fazendo uma predição

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [1705833000, 0, 12, 4]}'
```

`features` é `[datetime_epoch_seconds, cash_type_encoded, card_encoded, coffee_name_encoded]` — as mesmas quatro colunas e codificação que `src/utils/data_loader.py` produz no treino. Essa API ainda não aceita strings categóricas cruas; veja o artigo pra entender o porquê.

Rodando com Docker:

```bash
make docker-build
make docker-run
```

## Desenvolvimento

```bash
make test        # pytest
make coverage     # pytest com relatório de cobertura
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy
```

O CI roda ruff, pytest (com piso de cobertura), mypy e pip-audit em todo push/PR, além de uma execução diária agendada. Imagens Docker são construídas, escaneadas com Trivy e publicadas no GHCR na `main`. O Dependabot mantém pip, imagem base do Docker e GitHub Actions atualizados, com bumps patch/minor mesclados automaticamente. Releases são versionados automaticamente com [python-semantic-release](https://python-semantic-release.readthedocs.io/).
