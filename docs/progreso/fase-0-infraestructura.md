# Fase 0: Infraestructura del proyecto

**Fecha**: 2026-09-01
**Estado**: Completada

## Tareas completadas

### T-0.1: Crear estructura de directorios
- **Archivos creados**: `src/`, `data/raw/`, `data/processed/`, `docs/`, `tests/`, `notebooks/`, `config/`
- **Archivos creados**: `src/__init__.py`, `src/facial_attributes/__init__.py`, `tests/__init__.py`
- **Archivos creados**: `data/README.md`, `config/README.md`, `.gitkeep` files

### T-0.2: Configurar `pyproject.toml`
- **Archivos creados**: `pyproject.toml`
- **Dependencias principales**: torch, torchvision, scikit-learn, numpy, pandas, opencv-python, pillow, mlflow
- **Dependencias de desarrollo**: pytest, ruff, black, mypy, pre-commit
- **Configuración de herramientas**: ruff, black, mypy, pytest, coverage

### T-0.3: Crear `Makefile`
- **Archivos creados**: `Makefile`
- **Comandos disponibles**: install, sync, test, lint, format, typecheck, pre-commit, clean, validate

### T-0.4: Configurar `.gitignore`
- **Archivos creados**: `.gitignore`
- **Exclusiones**: datos raw, modelos, artefactos MLflow, secretos, archivos temporales, IDEs

### T-0.5: Configurar `pre-commit`
- **Archivos creados**: `.pre-commit-config.yaml`
- **Hooks configurados**: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, ruff, black, mypy

### T-0.6: Configurar CI/CD básico
- **Archivos creados**: `.github/workflows/ci.yml`
- **Jobs configurados**: lint, format, typecheck, test, pre-commit
- **Python versions**: 3.11, 3.12

## Archivos modificados/creados

```
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── config/
│   ├── README.md
│   └── .gitkeep (implícito)
├── data/
│   ├── README.md
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       ├── annotations/
│       │   └── .gitkeep
│       └── images/
│           └── .gitkeep
├── Makefile
├── notebooks/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   └── facial_attributes/
│       └── __init__.py
└── tests/
    └── __init__.py
```

## Decisiones técnicas

- **Build system**: hatchling (moderno y rápido)
- **Package manager**: uv (requerido por especificación)
- **Linting**: ruff (reemplaza flake8, isort, etc.)
- **Formatting**: black (estándar de la industria)
- **Type checking**: mypy (estricto)
- **Testing**: pytest (estándar)
- **CI/CD**: GitHub Actions (requerido por especificación)

## Referencia

- Especificación: `docs/specs.md` §16 (Constraints)
- Constitución: `docs/constitution.md` §3 (Calidad de software)
