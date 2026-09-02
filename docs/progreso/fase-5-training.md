# Fase 5: Training

## Estado: Completada

## Tareas completadas

### T-5.1: Configuración reproducible
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/training/config.py`
- **Descripción**:
  - Configuración con dataclasses
  - Soporte para semillas reproducibles
  - Detección automática de dispositivo (CPU/CUDA/MPS)
  - Creación automática de directorios

### T-5.2: Pipeline de entrenamiento
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/training/trainer.py`
  - `src/facial_attributes/training/dataset.py`
- **Descripción**:
  - Dataset personalizado para atributos faciales
  - Entrenamiento con early stopping
  - Validación por época
  - Predicciones y evaluación

### T-5.3: Tracking con MLflow
- **Estado**: Completada
- **Descripción**:
  - Integración con MLflow
  - Registro de parámetros y métricas
  - Comparación de experimentos

### T-5.4: Sistema de checkpoints
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/training/checkpoint.py`
- **Descripción**:
  - Guardado de checkpoints completos
  - Soporte para reanudar entrenamiento
  - Mantenimiento del mejor modelo
  - Metadatos en JSON

### T-5.5: Métricas de entrenamiento
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/training/metrics.py`
- **Descripción**:
  - Métricas multilabel (accuracy, precision, recall, F1)
  - Métricas por atributo
  - Cálculo de pérdida BCE

### T-5.6: Tests de training
- **Estado**: Completada
- **Archivos creados**:
  - `tests/test_training.py`
- **Descripción**: 23 tests cubriendo:
  - Configuración de entrenamiento
  - Reproducibilidad con semillas
  - Early stopping
  - Checkpoints
  - Dataset
  - Métricas
  - Trainer

## Archivos creados

```
src/facial_attributes/training/
├── __init__.py
├── config.py
├── dataset.py
├── checkpoint.py
├── metrics.py
└── trainer.py

tests/
└── test_training.py
```

## Uso del Trainer

```python
from facial_attributes.training import Trainer
from facial_attributes.training.config import TrainingConfig

config = TrainingConfig(
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    backbone="resnet18",
)

trainer = Trainer(config)
trainer.setup_model(num_attributes=40, pos_weight=class_weights)

history = trainer.train(train_loader, val_loader)
```

## Verificación

- 84/84 tests pasan
- Ruff check: OK
- Black format: OK

## Próximos pasos

- Fase 6: Evaluation (métricas y análisis de errores)
