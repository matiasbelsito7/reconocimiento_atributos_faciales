# Fase 11: Retraining Pipeline

## Estado: Completada

## Objetivo
Implementar pipeline de reentrenamiento para actualización incremental de modelos con nuevos datos.

## Componentes implementados

### 1. DatasetMerger (`src/facial_attributes/retraining/merger.py`)
- **MergeResult**: Dataclass con resultado de combinación
- **DatasetMerger**: Combinador controlado de datasets
  - `merge_datasets()`: Combina datasets existentes con nuevos
  - `validate_new_data()`: Valida nuevos datos antes de combinación
  - `_validate_schema()`: Valida compatibilidad de schemas
  - `_copy_new_images()`: Copia nuevas imágenes al directorio objetivo

### 2. AcceptanceCriteria (`src/facial_attributes/retraining/criteria.py`)
- **CriteriaResult**: Dataclass con resultado de verificación
- **AcceptanceCriteria**: Criterios de aceptación para reentrenamiento
  - `check_acceptance()`: Verifica si nuevo modelo cumple criterios
  - `compare_models()`: Compara métricas de dos modelos
  - Soporte para métricas donde menor es mejor (hamming_loss)

### 3. RetrainingPipeline (`src/facial_attributes/retraining/pipeline.py`)
- **RetrainingConfig**: Configuración del pipeline
- **RetrainingStep**: Paso del pipeline con estado y duración
- **RetrainingResult**: Resultado completo del pipeline
- **RetrainingPipeline**: Orquestador principal
  - `run()`: Ejecuta pipeline completo
  - Validación → Merge → Reentrenamiento → Evaluación → Registro

## Tests
- 23 tests en `tests/test_retraining.py`
- Cobertura completa de todos los componentes

## Archivos creados/modificados
- `src/facial_attributes/retraining/__init__.py`
- `src/facial_attributes/retraining/merger.py`
- `src/facial_attributes/retraining/criteria.py`
- `src/facial_attributes/retraining/pipeline.py`
- `tests/test_retraining.py`
- `docs/tasks.md` (actualizado)

## Validación
- ✅ 211 tests pasando
- ✅ ruff check: All checks passed
- ✅ black: Formato correcto

## Referencia
- **Especificación**: specs §14 (Retraining Pipeline)
- **Fecha**: 2026-09-02
