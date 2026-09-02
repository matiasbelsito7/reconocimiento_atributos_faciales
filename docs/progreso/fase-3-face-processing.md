# Fase 3: Face Processing

## Estado: Completada

## Tareas completadas

### T-3.1: Seleccionar detector de rostros
- **Estado**: Completada
- **Decisión**: OpenCV DNN con modelo Caffe pre-entrenado
- **Justificación**: Ya incluido en opencv-python, sin dependencias adicionales, buen rendimiento

### T-3.2: Implementar detección de rostros
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/face_processing/__init__.py`
  - `src/facial_attributes/face_processing/detector.py`
- **Descripción**: 
  - Detección de rostros usando OpenCV DNN
  - Modelo Caffe pre-entrenado (res10_300x300_ssd_iter_140000)
  - Soporte para múltiples rostros por imagen
  - Configuración de confianza y NMS

### T-3.3: Implementar extracción de rostros
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/face_processing/extractor.py`
- **Descripción**:
  - Extracción de rostros con margen configurable
  - Selección del rostro más grande
  - Metadata de posición y tamaño

### T-3.4: Implementar normalización
- **Estado**: Completada
- **Archivos creados**:
  - `src/facial_attributes/face_processing/normalizer.py`
- **Descripción**:
  - Redimensionamiento a tamaño uniforme (224x224)
  - Normalización de píxeles a [0, 1]
  - Soporte para procesamiento por lotes

### T-3.5: Manejo de errores en face processing
- **Estado**: Completada
- **Descripción**:
  - Error claro cuando modelo no está disponible
  - Manejo de imágenes sin rostros
  - Filtro de confianza mínima

### T-3.6: Tests de face processing
- **Estado**: Completada
- **Archivos creados**:
  - `tests/test_face_processing.py`
- **Descripción**: 20 tests cubriendo:
  - BoundingBox y DetectionResult
  - FaceDetector (inicialización, config, errores)
  - FaceExtractor (extracción, selección)
  - FaceNormalizer (normalización, batch)
  - FaceProcessingPipeline (integración)
  - ProcessedFace

## Archivos creados

```
src/facial_attributes/face_processing/
├── __init__.py
├── detector.py
├── extractor.py
├── normalizer.py
└── pipeline.py

tests/
└── test_face_processing.py
```

## Configuración del modelo

Para usar el detector, descargar los siguientes archivos en `models/face_detection/`:

1. `deploy.prototxt`: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
2. `res10_300x300_ssd_iter_140000.caffemodel`: https://dl.opencv.org/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel

## Verificación

- 46/46 tests pasan
- Ruff check: OK
- Black format: OK

## Próximos pasos

- Fase 4: Model (arquitectura y función de pérdida)
