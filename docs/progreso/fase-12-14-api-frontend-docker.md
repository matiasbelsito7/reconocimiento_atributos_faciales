# Fase 12-14: API, Frontend y Docker

**Fecha**: 2026-09-03
**Estado**: Completada

---

## Resumen

Implementación de la capa de servicio web completa: API REST con FastAPI, frontend web con cámara web y subida de archivos, y containerización con Docker.

---

## Archivos creados

### API (`src/facial_attributes/api/`)
- `__init__.py` — Módulo de API
- `schemas.py` — Modelos Pydantic: `PredictResponse`, `FaceResult`, `BoundingBoxResponse`, `HealthResponse`, `AttributesListResponse`, `AttributeInfo`
- `dependencies.py` — Singleton `InferencePipeline`, lista de 40 atributos CelebA con nombres display en español
- `routes.py` — Endpoints: `GET /api/health`, `POST /api/predict`, `GET /api/attributes`
- `main.py` — App FastAPI con lifespan manager, CORS middleware

### Frontend (`frontend/`)
- `index.html` — Layout responsive con dos paneles: entrada (webcam/upload) y resultados
- `css/style.css` — Diseño dark theme, responsive, cards de atributos con color coding
- `js/app.js` — Orquestador principal: health check, atributos, predicción, renderizado
- `js/webcam.js` — getUserMedia, captura manual, stop camera
- `js/upload.js` — Drag & drop, validación de tipo/tamaño, preview
- `nginx.conf` — Proxy inverso `/api/` → backend:8000, gzip, cache headers
- `Dockerfile` — nginx:alpine

### Docker (raíz)
- `Dockerfile` — Multi-stage (builder + runtime), PyTorch CPU-only, usuario no-root, healthcheck
- `.dockerignore` — Excluye data, models, tests, docs, .git
- `docker-compose.yml` — Servicios backend (8000) y frontend (3000), healthcheck, volumes para modelos

### Tests
- `tests/test_api.py` — 18 tests: health, attributes, predict, schemas, dependencies

### Archivos modificados
- `pyproject.toml` — Dependencias: fastapi, uvicorn, gunicorn, python-multipart, httpx (dev)
- `Makefile` — Targets Docker actualizados: docker-build, docker-run, docker-stop, docker-logs
- `docs/tasks.md` — 15 nuevas tareas documentadas (T-12.1 a T-14.5)

---

## Decisiones técnicas

| Decisión | Elección | Razón |
|----------|----------|-------|
| Framework API | FastAPI | Async, type hints, auto docs, dominante en ML serving 2025-2026 |
| Frontend | HTML/CSS/JS vanilla | Sin build step, sin dependencias, ligero |
| Webcam | getUserMedia + captura manual | Estándar del navegador, UX simple |
| Proxy | nginx en container separado | Resuelve CORS internamente en Docker network |
| Modelos | Volumes desde host | Imagen ligera (~500MB), modelos flexibles |
| Docker | Multi-stage build | 60-85% más pequeño que single-stage |
| PyTorch | CPU-only wheels | Reduce tamaño de imagen de ~700MB a ~170MB |

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/health` | Estado del servicio |
| POST | `/api/predict` | Predicción de atributos faciales |
| GET | `/api/attributes` | Lista de 40 atributos disponibles |

---

## Para probar localmente (sin Docker)

```bash
uv run uvicorn facial_attributes.api.main:app --reload --port 8000
```

Luego abrir `frontend/index.html` en el navegador.

## Para probar con Docker

```bash
docker compose build
docker compose up -d
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
```

---

## Test de integración Docker (T-14.5)

Se agregó `scripts/docker_smoke.py`, un smoke test ejecutable y repetible que:

1. Verifica que `docker compose build` construye ambas imágenes.
2. Levanta backend y frontend con `docker compose up -d` (proyecto aislado `facial-attributes-smoke`).
3. Espera que `GET /api/health` responda `status=ok` (funciona sin pesos del modelo).
4. Verifica que el frontend responde HTTP 200.
5. Detiene los servicios al finalizar (siempre, incluso en fallo).

Accesible vía `make docker-test` y como job `docker-smoke` en `.github/workflows/ci.yml`.

**Corrección al `Dockerfile`**: el runtime no copiaba `pyproject.toml`/`README.md`, por lo que el `pip install -e .` fallaba cualquier build. Ahora la aplicación se empaqueta como wheel en el builder (`pip wheel --no-deps .`) y se instala desde `/wheels` junto al resto de dependencias; se eliminó el `COPY src/` redundante y el `pip install -e .` del runtime.

---

## Pendiente

- **Modelo entrenado**: La API funciona pero sin modelo entrenado, `/predict` devuelve error "Modelo no cargado"
- **Pesos face detector**: `res10_300x300_ssd_iter_140000.caffemodel` debe descargarse manualmente
