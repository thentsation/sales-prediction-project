🇧🇷 Português | [🇺🇸 English](ARTIGO.en-us.md)

# A API que nunca escalava as features na hora de prever

Esse foi o bug mais sério que encontrei em toda essa rodada de produtização, e o mais silencioso: a API respondia 200, devolvia um número, e esse número estava sistematicamente errado.

## O que eu tinha em mãos

Um pipeline de previsão de vendas — `RandomForestRegressor` com `StandardScaler`, treino via `train.py`, tracking com MLflow, serving via FastAPI — numa estrutura duplicada entre `api/`, `src/models/`, `src/utils/` na raiz. `data_loader.py` engolia toda exceção e devolvia `None` silenciosamente em vários pontos, o que na prática não protegia nada: `X_train, X_test, y_train, y_test = load_data(...)` não tem como desempacotar `None` em quatro variáveis, então o "tratamento de erro" só adiava o crash pra um `TypeError` sem contexto na linha seguinte.

## O bug real: `save_model` salvava o modelo, não o preditor

```python
# original
def save_model(self, path="models/sales_predictor.pkl"):
    joblib.dump(self.model, path)
```

`self.model` é só o `RandomForestRegressor` interno. O `StandardScaler` — ajustado durante `train()`, usado dentro de `predict()` pra escalar a entrada antes de prever — nunca era salvo. A API carregava esse pickle e chamava `.predict()` **direto no RandomForest**, pulando o scaler inteiro:

```python
# api/main.py original
model = joblib.load("models/sales_predictor.pkl")
...
prediction = model.predict(np.array([request.features]))[0]
```

Ou seja: toda predição em produção usava features na escala original (não escalada), enquanto o modelo foi treinado com features escaladas. Sem erro, sem exceção — só um número sistematicamente deslocado, porque `RandomForestRegressor` não é tão sensível a escala quanto modelos lineares, mas ainda assim se comporta de forma diferente com magnitudes de entrada distintas das de treino. Isso não aparece em nenhum log, nenhum teste, nenhum health check. Só aparece quando alguém compara a predição com o valor real.

A correção foi trocar o que é serializado: em vez de `self.model`, `save_model` agora serializa **o `SalesPredictor` inteiro** (que carrega `model` e `scaler` juntos), e `load()` devolve essa mesma instância:

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

O teste que teria pego isso — e que escrevi assim que entendi o problema — treina, salva, recarrega e compara a predição do objeto recarregado com a do original:

```python
def test_save_and_load_roundtrip_preserves_scaling(tmp_path) -> None:
    predictor = SalesPredictor()
    predictor.train(X, y)
    expected = predictor.predict(X)

    predictor.save_model(str(tmp_path / "model.pkl"))
    loaded = SalesPredictor.load(str(tmp_path / "model.pkl"))

    np.testing.assert_array_almost_equal(loaded.predict(X), expected)
```

## `logging.info` que nunca logava nada

`train.py` chamava `logging.info("Starting training!")` no nível de módulo, sem nunca configurar o logging (`logging.basicConfig`). Sem configuração, o logger raiz do Python usa um handler que descarta tudo abaixo de `WARNING` — ou seja, essas chamadas de `info` nunca apareciam em lugar nenhum. Troquei por um `setup_logging()` de verdade (o mesmo padrão dos outros projetos da organização) e passei o logger explicitamente pro `SalesPredictor`, trocando os `print()` espalhados por chamadas de log de verdade.

## Erros explícitos em vez de `None` silencioso

Reescrevi `data_loader.load_data` pra levantar `ValueError` com mensagem clara em cada caso que antes devolvia `None` — arquivo vazio, coluna faltando, tudo vazio depois do `dropna`. Um `ValueError("Missing expected column(s): [...]")` na hora certa é infinitamente mais fácil de debugar do que um `TypeError` genérico três frames depois.

## O que não mudei

O contrato da API continua esperando features já codificadas (`[datetime_epoch, cash_type_encoded, card_encoded, coffee_name_encoded]`) em vez de aceitar strings categóricas cru. Os `LabelEncoder`s usados no treino não são persistidos nem versionados — quem chama a API precisa saber a codificação de antemão. Não redesenhei esse contrato: é uma limitação real do projeto original, documentada agora no README, mas mudar a forma como a API recebe input é uma decisão de produto que não me cabia tomar sozinho nessa passada — o bug que valia a pena consertar sem pedir permissão era o scaler nunca aplicado, não o design da API.
