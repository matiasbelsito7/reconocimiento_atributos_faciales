# Config

Configuración del sistema en archivos YAML, consumida por el módulo Python `src/facial_attributes/config/`.

## Archivos YAML

- `pipeline.yaml` - Orquestación global del pipeline: modo (`training`/`inference`), paths, logging, seed y device.
- `model.yaml` - Arquitectura del modelo: backbone, tamaño de entrada/salida y regularización (dropout).
- `training.yaml` - Entrenamiento: hiperparámetros, early stopping, checkpoints, augmentation y tracking (MLflow).
- `inference.yaml` - Inferencia: thresholds por atributo, detección/extracción facial, márgenes de incerteza y formato de salida.
- `datasets.yaml` - Definición de datasets fuente (`celeba`, `sample`): paths, imágenes, atributos, splits y validación.

## Módulo Python (`src/facial_attributes/config/`)

- `schemas.py` - Esquemas tipados (dataclasses) de todas las secciones: `PipelineConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig`, `DatasetsConfig` y sus sub-configs. Definen los campos válidos y sus valores por defecto.
- `loader.py` - Clase `ConfigLoader`: lee los YAML del directorio `config/` (relativo al CWD; `WORKDIR /app` en el contenedor) y construye las dataclasses de `schemas.py`. Expone `load_pipeline()`, `load_model()`, `load_training()`, `load_inference()`, `load_datasets()` y `load_all()`.
- `__init__.py` - Reexporta `ConfigLoader` y las configuraciones principales para consumo externo.
