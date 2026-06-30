.PHONY: install lint format test test-cov train run-api mlflow-ui clean help

PYTHON := python3
SRC := src
TESTS := tests

help:
	@echo "Comandos disponíveis:"
	@echo "  make install     - instala o projeto e dependências de desenvolvimento"
	@echo "  make lint        - roda ruff (verificação de estilo e erros)"
	@echo "  make format      - aplica autofix do ruff"
	@echo "  make test        - roda a suíte de testes (pytest)"
	@echo "  make test-cov    - roda os testes com relatório de cobertura"
	@echo "  make train       - executa o pipeline de treino completo (baselines + MLP)"
	@echo "  make run-api     - inicia a API FastAPI localmente (porta 8000)"
	@echo "  make mlflow-ui   - abre a interface web do MLflow"
	@echo "  make clean       - remove caches e artefatos temporários"

install:
	pip install -e ".[dev]"

lint:
	ruff check $(SRC) $(TESTS)

format:
	ruff check --fix $(SRC) $(TESTS)
	ruff format $(SRC) $(TESTS)

test:
	pytest $(TESTS) -v

test-cov:
	pytest $(TESTS) --cov=churn_prediction --cov-report=term-missing

train:
	$(PYTHON) -m churn_prediction.train

run-api:
	uvicorn churn_prediction.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
