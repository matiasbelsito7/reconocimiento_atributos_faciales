# AGENTS.md

## # Proyecto

**Facial Attribute Recognition**

Proyecto de Computer Vision / Data Science para reconocimiento multilabel de atributos faciales visualmente observables mediante modelos de Machine Learning y Deep Learning.

El sistema evolucionará hacia un pipeline reproducible de:

```text
data → data_validation → preprocessing → face processing → training
                                                              ↓
                                                       evaluation
                                                              ↓
                                                       inference
                                                              ↓
                                                       monitoring
```

Stack principal:

* Python 3.11+
* PyTorch
* torchvision
* scikit-learn
* NumPy / pandas
* OpenCV / Pillow
* MLflow
* pytest
* Ruff
* Black
* mypy
* pre-commit
* uv
* Docker
* GitHub Actions

La arquitectura y especificaciones detalladas se encuentran en `docs/specs.md`.

Los principios que gobiernan las decisiones técnicas se encuentran en `docs/constitution.md`.

---

## # Comandos

Preferir siempre los comandos definidos en el `Makefile` cuando existan.

Comandos base:

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run black .
uv run mypy .
uv run pre-commit run --all-files
```

El comando oficial para ejecutar la aplicación debe estar documentado en el `Makefile`.

---

## # Estilo y convenciones

* Usar Python con type hints.
* Mantener funciones y módulos con responsabilidades claras.
* Seguir las reglas de Ruff y Black.
* Evitar duplicación y abstracciones innecesarias.
* No hardcodear secretos ni paths dependientes de una máquina.
* Mantener separadas las etapas de datos, preprocessing, entrenamiento, evaluación e inferencia.
* Utilizar notebooks para exploración y prototipado; la lógica reutilizable debe vivir en `src/`.
* Agregar tests para cambios relevantes.
* No modificar datos raw manualmente.
* Usar Conventional Commits.

Las convenciones detalladas y decisiones permanentes pertenecen a `docs/constitution.md`.

---

## # Reglas

### SDD

El proyecto utiliza **Specification-Driven Development**.

Antes de implementar una tarea, el agente debe consultar:

1. `docs/constitution.md`
2. `docs/specs.md`
3. `docs/tasks.md`

Luego debe inspeccionar el código existente relevante.

### Alcance

Implementar únicamente lo definido por la tarea y sus especificaciones.

No introducir funcionalidades, dependencias o cambios arquitectónicos no justificados.

### Seguridad

No incluir secretos, credenciales ni información personal innecesaria en el repositorio.

No implementar inferencias de atributos sensibles a partir de rostros.

### Calidad

No ocultar errores para hacer pasar tests, linters o CI.

No eliminar ni debilitar tests o herramientas de calidad sin justificación.

### Documentación

Cada documento tiene una responsabilidad específica:

* `AGENTS.md` → comportamiento y contexto operativo de agentes.
* `docs/constitution.md` → principios.
* `docs/specs.md` → especificaciones.
* `docs/tasks.md` → tareas.
* `docs/progreso/` → registro de tareas completadas.
* `skills/*/SKILL.md` → procedimientos y capacidades especializadas.

No duplicar contenido entre estos documentos.

### Skills

Los skills definen procedimientos reutilizables para operaciones específicas. Consultar el skill correspondiente antes de ejecutar una operación:

* `skills/development-workflow/SKILL.md` → workflow general de desarrollo
* `skills/data-validation/SKILL.md` → validación de datasets
* `skills/dataset-management/SKILL.md` → gestión de datasets
* `skills/model-training/SKILL.md` → entrenamiento de modelos
* `skills/experiment-tracking/SKILL.md` → registro de experimentos
* `skills/model-evaluation/SKILL.md` → evaluación de modelos
* `skills/inference-pipeline/SKILL.md` → pipeline de inferencia
* `skills/code-quality/SKILL.md` → verificación de calidad de código

---

## # Al terminar una tarea

1. Documentar la tarea completada en `docs/progreso/`:
   * Nombre de la tarea (ej: `T-0.1`).
   * Descripción breve de lo implementado.
   * Archivos modificados o creados.
   * Decisiones técnicas tomadas (si las hubo).
   * Referencia a la especificación utilizada.

2. Seguir el workflow de finalización:

`skills/development-workflow/SKILL.md`

El workflow de validación, commit, push, CI y verificación no se define aquí.
