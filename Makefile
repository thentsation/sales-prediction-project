.PHONY: install train serve test coverage lint format typecheck docker-build docker-run clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r config/requirements.txt
	$(PIP) install -r config/requirements-dev.txt

train:
	PYTHONPATH=src $(PYTHON) src/main.py

serve:
	PYTHONPATH=src $(VENV)/bin/uvicorn api.main:app --reload

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/ruff format .

typecheck:
	$(VENV)/bin/mypy

docker-build:
	docker build -f docker/Dockerfile -t sales-prediction-project .

docker-run:
	docker run --rm -p 8021:8000 sales-prediction-project

clean:
	find . -type d -name __pycache__ -not -path './$(VENV)/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -not -path './$(VENV)/*' -exec rm -rf {} +
	find . -type f -name '*.pyc' -not -path './$(VENV)/*' -delete
	rm -f .coverage
