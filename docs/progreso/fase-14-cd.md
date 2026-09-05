# T-14.6: Workflow de CD para despliegue del modelo champion

**Fecha**: 2026-09-05
**Estado**: Completada

---

## Descripción

Creación del pipeline de Continuous Deployment que se ejecuta cada vez que cambia el modelo champion (estado `PRODUCTION` en el Model Registry). Construye las imágenes Docker del backend y frontend, las publica en GitHub Container Registry y despliega en el servidor de producción vía SSH con los pesos del nuevo champion.

---

## Archivos creados

- `.github/workflows/cd.yml` — Workflow CD con dos jobs: `build-and-push` (GHCR) y `deploy` (SSH + Docker Compose)
- `docker-compose.prod.yml` — Compose de producción basado en imágenes GHCR parametrizadas por `BACKEND_IMAGE`, `FRONTEND_IMAGE` e `IMAGE_TAG`

## Archivos modificados

- `docs/tasks.md` — Tarea T-14.6 documentada + decisiones de CD agregadas al resumen
- `docs/specs.md` — Nueva sección §13.6 "Continuous Deployment del modelo champion"

---

## Decisiones técnicas

| Decisión | Elección | Razón |
|----------|----------|-------|
| Trigger | `repository_dispatch` (`champion-model-changed`) + fallback `workflow_dispatch` | El pipeline de entrenamiento puede notificar al promocionar el champion; manual como respaldo |
| Registro de imágenes | GitHub Container Registry (`ghcr.io/{repo}/backend` y `/frontend`) | Sin infraestructura extra, token de Actions incluido |
| Tags | `latest` + versión del champion sanitizada | Trazabilidad de qué imagen corresponde a qué champion |
| Deploy | SSH (secrets `CD_HOST`, `CD_USERNAME`, `CD_SSH_PRIVATE_KEY`) + `docker compose -f docker-compose.prod.yml pull/up` | Sin dependencias de agentes cloud; reutiliza el stack Docker existente |
| Pesos del champion | Descarga por URL (`model_url`) a `checkpoints/best_model.pt` en el servidor, con escritura atómica (`.new` + `mv`) | La imagen no incluye modelos (se mantiene liviana); el volumen es de solo lectura |
| Verificación | Health check remoto verificando `model_loaded: true` | Confirma que el champion quedó cargado, no solo que el contenedor está arriba |
| Concurrencia | `concurrency: group: cd` con `cancel-in-progress: false` | Evita que dos deploys se cancelen entre sí |

---

## Secrets y configuración requeridos

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `CD_HOST` | Secret | Host/IP del servidor de producción |
| `CD_USERNAME` | Secret | Usuario SSH |
| `CD_SSH_PRIVATE_KEY` | Secret | Clave privada SSH |
| `CD_DEPLOY_PATH` | Variable (opcional) | Directorio de deploy en el servidor (default `/opt/facial-attributes`) |

---

## Referencia a especificación

- `docs/specs.md` §13 (Model Registry), §13.6 (Continuous Deployment del modelo champion), §16 (Constraints)
- `docs/constitution.md` §3 (Calidad de software: CI/CD funcional)

---

## Pendiente

- **Receptor del `repository_dispatch`**: el pipeline de entrenamiento/evaluación debe emitir el evento al promocionar un modelo a `PRODUCTION` (payload documentado en `docs/tasks.md` T-14.6).
