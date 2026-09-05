.PHONY: help install sync test lint format typecheck pre-commit clean

help: ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instalar dependencias del proyecto
	uv sync --all-groups

sync: ## Sincronizar dependencias
	uv sync --all-groups

test: ## Ejecutar tests
	uv run pytest

test-cov: ## Ejecutar tests con cobertura
	uv run pytest --cov=facial_attributes --cov-report=html

lint: ## Verificar código con Ruff
	uv run ruff check .

lint-fix: ## Corregir errores de lint
	uv run ruff check --fix .

format: ## Formatear código con Black
	uv run black .

format-check: ## Verificar formateo
	uv run black --check .

typecheck: ## Verificar tipos con mypy
	uv run mypy src/

pre-commit: ## Ejecutar pre-commit en todos los archivos
	uv run pre-commit run --all-files

clean: ## Limpiar archivos temporales
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache htmlcov .coverage

validate: lint format-check typecheck test ## Ejecutar todas las validaciones

docker-build: ## Construir imagenes Docker
	docker compose build

docker-run: ## Ejecutar servicios Docker
	docker compose up -d

docker-stop: ## Detener servicios Docker
	docker compose down

docker-logs: ## Ver logs de Docker
	docker compose logs -f

run-api: ## Ejecutar API en desarrollo (uvicorn --reload)
	uv run uvicorn facial_attributes.api.main:app --reload --port 8000

run-app: ## Ejecutar API + Frontend en http://localhost:8000
	uv run uvicorn facial_attributes.api.main:app --reload --port 8000

run-api-prod: ## Ejecutar API en produccion (gunicorn)
	uv run gunicorn facial_attributes.api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
