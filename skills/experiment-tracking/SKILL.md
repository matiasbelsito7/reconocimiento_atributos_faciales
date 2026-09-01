# Experiment Tracking

## Purpose

Registrar y comparar experimentos de entrenamiento utilizando MLflow.

## When to use

- Durante el entrenamiento de modelos
- Al comparar experimentos
- Al revisar historial de experimentos

## What to register

### Per experiment

- **Configuración**: hiperparámetros, semilla, versión de datos
- **Métricas**: loss, accuracy, F1, etc. por época
- **Artefactos**: modelo entrenado, gráficas, logs
- **Duración**: tiempo total de entrenamiento

### Tags

- Nombre descriptivo del experimento
- Versión del código
- Dataset utilizado
- Arquitectura del modelo

## Procedure

### 1. Initialize

```python
import mlflow

mlflow.set_experiment("facial-attribute-recognition")
```

### 2. Log during training

```python
with mlflow.start_run(run_name="run-name"):
    mlflow.log_params(config)
    mlflow.log_metric("train_loss", epoch_loss)
    mlflow.log_metric("val_f1", f1_score)
    mlflow.save_model(model, "model")
```

### 3. Compare

- Abrir MLflow UI para comparar runs
- Identificar mejor configuración
- Revisar métricas y parámetros

## Output

- Registro persistente de cada experimento
- Capacidad de comparación entre runs
- Trazabilidad de decisiones de entrenamiento

## Related

- `docs/specs.md` §6.4 (Tracking de experimentos)
- `docs/constitution.md` §2 (ML Principles)
- `skills/model-training/SKILL.md`
