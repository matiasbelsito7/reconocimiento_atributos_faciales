# Fase 7: Inference

## Estado: Completada

## Tareas completadas

### T-7.1: Pipeline de inferencia
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/inference/__init__.py`
  - `src/facial_attributes/inference/pipeline.py`
- **Descripción**:
  - Pipeline completo: imagen → preprocessing → face detection → prediction → output
  - Integración con módulos de face processing y model
  - Soporte para inferencia individual y por lotes
  - Detección automática de dispositivo (CPU/CUDA/MPS)

### T-7.2: Formato de salida
- **Estado**: Completada
- **Descripción**:
  - Estructura de datos FacePrediction con bbox y attributes
  - InferenceResult con metadata completa
  - Formato JSON compatible con specs

### T-7.3: Manejo de errores en inferencia
- **Estado**: Completada
- **Descripción**:
  - Error cuando modelo no está cargado
  - Error cuando no se detectan rostros
  - Manejo de excepciones con mensajes claros
  - No falla silenciosamente

### T-7.4: Tests de inferencia
- **Estado**: Completada
- **Archivos creados**:
  - `tests/test_inference.py`
- **Descripción**: 12 tests cubriendo:
  - Configuración de inferencia
  - FacePrediction e InferenceResult
  - InferencePipeline (inicialización, predicción, batch)
  - Integración con modelo mockeado
  - Estructura de resultados

## Archivos creados

```
src/facial_attributes/inference/
├── __init__.py
└── pipeline.py

tests/
└── test_inference.py
```

## Uso del InferencePipeline

```python
from facial_attributes.inference import InferencePipeline
from facial_attributes.inference.pipeline import InferenceConfig

config = InferenceConfig(
    model_path="checkpoints/best_model.pt",
    threshold=0.5,
    num_attributes=40,
    attribute_names=["smiling", "glasses", ...],
)

pipeline = InferencePipeline(config)
result = pipeline.predict_from_path(Path("image.jpg"))

print(f"Rostros detectados: {result.num_faces_detected}")
for face in result.faces:
    print(f"  BBox: {face.bbox}")
    print(f"  Atributos: {face.attributes}")
```

## Formato de salida

```json
{
  "faces": [
    {
      "bbox": {"x": 100, "y": 80, "w": 100, "h": 140},
      "attributes": {
        "smiling": 0.92,
        "glasses": 0.15
      },
      "confidence": 0.95
    }
  ],
  "num_faces_detected": 1,
  "num_faces_with_predictions": 1,
  "inference_time_ms": 45.2,
  "image_size": [640, 480]
}
```

## Verificación

- 110/110 tests pasan
- Ruff check: OK
- Black format: OK

## Próximos pasos

- Fase 8: Configuration Module (archivos YAML)
