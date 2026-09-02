# Fase 4: Model

## Estado: Completada

## Tareas completadas

### T-4.1: Seleccionar arquitectura del modelo
- **Estado**: Completada
- **Decisión**: ResNet18/34/50 pre-entrenada en ImageNet
- **Justificación**: Buen balance entre precisión y eficiencia, transfer learning efectivo

### T-4.2: Implementar modelo
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/model/__init__.py`
  - `src/facial_attributes/model/classifier.py`
- **Descripción**:
  - Clasificador multilabel basado en ResNet
  - Soporte para ResNet18, ResNet34, ResNet50
  - Capa de clasificación personalizada con dropout
  - Métodos predict_proba() y predict()
  - Opción de congelar backbone para fine-tuning

### T-4.3: Seleccionar función de pérdida
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/model/losses.py`
- **Descripción**:
  - BCE Loss para clasificación multilabel
  - Soporte para pesos de clase (pos_weight)
  - Wrapper configurable

### T-4.4: Tests del modelo
- **Estado**: Completada
- **Archivos creados**:
  - `tests/test_model.py`
- **Descripción**: 15 tests cubriendo:
  - Configuración del modelo
  - Forward pass
  - Predicción de probabilidades
  - Predicción binaria
  - Conteo de parámetros
  - Diferentes backbones
  - Congelamiento de backbone
  - Función de pérdida

## Archivos creados

```
src/facial_attributes/model/
├── __init__.py
├── classifier.py
└── losses.py

tests/
└── test_model.py
```

## Configuración del modelo

```python
from facial_attributes.model import FacialAttributeClassifier, ModelConfig

config = ModelConfig(
    num_attributes=40,
    backbone="resnet18",
    pretrained=True,
    dropout_rate=0.5,
    freeze_backbone=False,
)
model = FacialAttributeClassifier(config)
```

## Verificación

- 61/61 tests pasan
- Ruff check: OK
- Black format: OK

## Próximos pasos

- Fase 5: Training (pipeline de entrenamiento con MLflow)
