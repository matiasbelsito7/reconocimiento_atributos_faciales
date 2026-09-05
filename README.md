# Facial Attribute Recognition

Reconocimiento **multilabel** de atributos faciales visualmente observables utilizando Machine Learning y Deep Learning.

El sistema predice simultáneamente la presencia o ausencia de múltiples atributos faciales (expresiones, accesorios, características visibles) a partir de imágenes.

> **Uso responsable**: el sistema trabaja **exclusivamente** con atributos faciales visualmente observables. Está prohibido utilizarlo para inferir atributos sensibles (raza, orientación sexual, discapacidad, estado de salud, afiliación política o religiosa). Ver [Constitución](docs/constitution.md#5-privacidad-y-uso-responsable).

---

## Características

- **Pipeline reproducible** `data → validation → preprocessing → face processing → training → evaluation → inference → monitoring` con separación estricta de etapas.
- **API REST** (FastAPI) con endpoints de salud, predicción y listado de atributos.
- **Frontend web** con captura por webcam y subida de archivos.
- **Containerización** con Docker (multi-stage, PyTorch CPU-only) y orquestación con Docker Compose.
- **Model Registry** con versionado, estados (development/staging/production/archived) e integración con MLflow.
- **Continuous Deployment** del modelo champion: al promocionar un modelo a producción, se reconstruye la imagen y se despliega el nuevo checkpoint automáticamente.
- **Entrenamiento en CPU** con cache de imágenes pre-procesadas y `pos_weight` automático.
- **Métricas de evaluación completas**: Precision, Recall, F1, Macro-F1, PR-AUC y ROC-AUC por atributo y globales.
- **Validation, evaluation y retraining** pipelines implementados (incluidos criterios de aceptación para reentrenamiento).

---

## Arquitectura

```text
data → data_validation → preprocessing → face processing → training
                                                              ↓
                                                       evaluation
                                                              ↓
                                                       inference
                                                              ↓
                                                       monitoring
```

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Configuration Module                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Management                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Validation                               │
└─────────────────────────────────────────────────────────────────┘
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│    Training Pipeline      │   │   Inference Pipeline       │
│  Preprocessing → Face →   │   │  Preprocessing → Face →    │
│  Training → Evaluation    │   │  Inference → Monitoring    │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│      Evaluation           │   │      Monitoring            │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Model Registry (MLflow)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack tecnológico

- **Python 3.11+**, **PyTorch** y **torchvision** (ResNet fine-tuning, BCE loss multilabel)
- **scikit-learn**, **NumPy**, **pandas**
- **OpenCV / Pillow** (procesamiento de imágenes y detección de rostros)
- **FastAPI / Uvicorn / Gunicorn** (API), **nginx** (frontend y proxy inverso)
- **MLflow** (tracking y Model Registry), **DVC** (versionado de datos)
- **Docker / Docker Compose**, **GitHub Actions** (CI/CD)
- Calidad: **pytest**, **Ruff**, **Black**, **mypy**, **pre-commit**

---

## Estructura del repositorio

```text
├── src/facial_attributes/
│   ├── api/               # Endpoints FastAPI (health, predict, attributes)
│   ├── config/            # Carga y validación de configuración YAML
│   ├── data/              # Gestión de datasets y validación de datos
│   ├── preprocessing/     # Pipelines training / inference
│   ├── face_processing/   # Detección y extracción de rostros
│   ├── model/             # Modelo multilabel y funciones de pérdida
│   ├── training/          # Entrenamiento, checkpoints, métricas
│   ├── evaluation/        # Evaluación por atributo y thresholds
│   ├── inference/         # Pipeline de inferencia
│   ├── monitoring/        # Métricas, logs y alertas en producción
│   ├── model_registry/    # Registro y versionado de modelos (MLflow)
│   └── retraining/        # Reentrenamiento y criterios de aceptación
├── frontend/              # Web (webcam/upload) + nginx
├── config/                # YAMLs de configuración centralizada
├── scripts/               # Scripts de datos y entrenamiento en CPU
├── data/                  # raw / processed / merged (gitignored)
├── docs/                  # Constitución, specs, tareas y progreso
├── tests/                 # Suite de pytest
├── skills/                # Skills/guías del equipo
├── Dockerfile             # Backend multi-stage (PyTorch CPU)
├── docker-compose.yml     # Desarrollo (build local)
├── docker-compose.prod.yml# Producción (imágenes GHCR)
└── .github/workflows/     # CI + CD
```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/matiasbelsito7/reconocimiento_atributos_faciales.git
cd reconocimiento_atributos_faciales

# Instalar dependencias
uv sync --all-groups

# Configurar pre-commit
uv run pre-commit install
```

---

## Uso

```bash
make validate        # Ruff + Black + mypy + pytest
make test            # Ejecutar tests
make lint            # Ruff
make format          # Black
make typecheck       # mypy
make run-api         # Levantar API (uvicorn --reload) en :8000
```

Todos los comandos disponibles: `make help`.

---

## API REST

Backend FastAPI en `http://localhost:8000` (docs interactivas en `/docs`).

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/health` | Estado del servicio, dispositivo, modelo cargado |
| POST | `/api/predict` | Predicción de atributos (multipart: `file`) |
| GET | `/api/attributes` | Lista de los 40 atributos soportados |

Ejemplo:

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@foto.jpg"
```

```json
{
  "faces": [
    {
      "bbox": { "x": 120, "y": 80, "w": 200, "h": 200 },
      "attributes": { "Smiling": 0.92, "Eyeglasses": 0.15, "Young": 0.8 },
      "confidence": 0.98
    }
  ],
  "num_faces_detected": 1,
  "num_faces_with_predictions": 1,
  "inference_time_ms": 45.2,
  "image_size": [640, 480],
  "error": null
}
```

Formatos soportados: JPEG, PNG, WebP, BMP (máx. 10 MB).

---

## Frontend

Frontend estático servido por nginx con proxy inverso hacia el backend:

```bash
# Sin Docker: abrir frontend/index.html en el navegador
# Con Docker: http://localhost:3000
```

Funcionalidades: captura con webcam (`getUserMedia`), subida de archivos con drag & drop, y renderizado de resultados (bounding box + scores por atributo) con color coding.

---

## Docker

### Desarrollo

```bash
make docker-build      # docker compose build
make docker-run        # docker compose up -d
make docker-stop       # docker compose down
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

Los checkpoints de modelos se montan como volúmenes de solo lectura (`./checkpoints` y `./models`).

### Producción y Continuous Deployment

El workflow de CD (`cd.yml`) se dispara cuando **cambia el modelo champion** (estado `production` del Model Registry, vía `repository_dispatch` `champion-model-changed`):

1. Construye y publica las imágenes `backend` y `frontend` en GitHub Container Registry (tags `latest` + versión del champion).
2. Descarga el checkpoint del champion al servidor (URL provista en el payload).
3. Recrea los servicios con `docker-compose.prod.yml`.
4. Verifica `model_loaded: true` en `/api/health`.

Secrets requeridos: `CD_HOST`, `CD_USERNAME`, `CD_SSH_PRIVATE_KEY`.

---

## Entrenamiento

### Pipeline de entrenamiento

```text
preprocessing_train → face processing → training → evaluation → model_registry
```

- **Loss**: `BCEWithLogitsLoss` con `pos_weight` automático (`num_neg / num_pos`) calculado desde el set de entrenamiento.
- **Checkpoints**: guardado por época + mejor modelo (`best_model.pt`), con reanudación.
- **Tracking**: cada experimento se registra en MLflow (config, métricas, artefactos, duración).
- **Métricas**: Precision, Recall, F1, Macro-F1, PR-AUC y ROC-AUC por atributo y globales.

### Entrenamiento en CPU (subconjuntos)

```bash
# Crear subconjunto estratificado (5000 muestras por defecto)
uv run python scripts/build_subset.py

# Pre-procesar imágenes a cache .npy (acelera el entrenamiento)
uv run python scripts/build_cache.py

# Entrenar sobre el subconjunto (ResNet18, 15 épocas por defecto)
uv run python scripts/train_subset.py
```

- `build_cache.py`: redimensiona y cachea imágenes como `.npy` para evitar re-procesar en cada época.
- `train_subset.py`: entrenador ligero para CPU → `checkpoints/best_model.pt`.
- `CachedAttributeDataset` salta el redimensionamiento en cada época (gran aceleración en CPU).

### Dataset CelebA

- Fuente, licencia y procedencia documentadas en `docs/progreso/fase-1-data.md`.
- Se descarga con `scripts/download_celeba.py` y se valida con `scripts/validate_data.py`.

---

## Model Registry y MLflow

Módulo `model_registry` con estados:

| Estado | Descripción |
|--------|-------------|
| `development` | En desarrollo, no listo para uso |
| `staging` | Evaluado, pendiente de validación final |
| `production` | **Champion**: validado y en producción |
| `archived` | Deprecado o reemplazado |

Operaciones soportadas: registrar modelos, comparar (ganador por F1), promocionar a producción, agregar artefactos, listar por estado y eliminar. `MLflowRegistry` integra el registro con MLflow (`log_training_run`, `transition_model_version`, `load_model`).

La promoción del **champion** a producción dispara el CD automático descrito arriba.

---

## CI/CD

- **CI** (`ci.yml`): Ruff, Black, mypy vía pre-commit, y pytest en Python 3.11 y 3.12 en cada push/PR a `main`/`develop`.
- **CD** (`cd.yml`): build → GHCR → deploy SSH al servidor al cambiar el modelo champion (ver sección Docker).

---

## Configuración

Configuración centralizada en `config/`:

| Archivo | Contenido |
|---------|-----------|
| `pipeline.yaml` | Rutas de directorios, modo, semilla, dispositivo |
| `model.yaml` | Arquitectura e hiperparámetros del modelo |
| `training.yaml` | Semilla, LR, batch size, épocas, early stopping |
| `inference.yaml` | Thresholds, face detection, optimización |
| `datasets.yaml` | Datasets disponibles, rutas y splits |

Ver `config/README.md` para la referencia detallada.

---

## Documentación

- [Constitución del proyecto](docs/constitution.md) — principios
- [Especificaciones](docs/specs.md) — estado deseado del sistema
- [Tareas](docs/tasks.md) — backlog y estado de implementación
- [Progreso](docs/progreso/README.md) — registro de tareas completadas

---

## Licencia

MIT

---

*Proyecto desarrollado bajo **Specification-Driven Development**. Ver `AGENTS.md` y `docs/constitution.md`.*
