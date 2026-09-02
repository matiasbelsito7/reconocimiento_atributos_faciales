# Development Workflow

## Purpose

Define el procedimiento estándar para implementar, validar y cerrar una tarea del proyecto.

## 1. Understand

Antes de modificar código:

1. Leer `AGENTS.md`.
2. Consultar la tarea correspondiente en `docs/tasks.md`.
3. Consultar las especificaciones relevantes en `docs/specs.md`.
4. Consultar la constitución cuando sea necesario.
5. Inspeccionar el código existente relacionado.

La implementación debe limitarse al alcance definido por la tarea.

## 2. Implement

Durante la implementación:

- mantener la arquitectura existente
- reutilizar código cuando corresponda
- agregar tests para cambios relevantes
- mantener type hints
- no introducir dependencias innecesarias
- no modificar configuración de calidad para ocultar errores

## 3. Validate locally

Ejecutar:

```bash
uv run pre-commit run --all-files
uv run pytest
```

El pre-commit debe incluir como mínimo:

- Ruff
- Black
- mypy
- pytest

Si alguna herramienta modifica archivos, revisar nuevamente el diff y repetir la validación.

Revisar finalmente:

```bash
git status
git diff
```

No deben quedar archivos temporales, secretos ni cambios no relacionados.

## 4. Commit

Crear un commit siguiendo Conventional Commits.

Ejemplos:

```text
feat: add facial attribute classifier
fix: correct image preprocessing
test: add inference tests
refactor: separate training pipeline
docs: update model evaluation
```

No usar mensajes genéricos.

## 5. Push

Después de validar localmente:

```bash
git push
```

## 6. CI

Verificar los workflows de GitHub Actions relacionados con los cambios.

Como mínimo:

- lint
- tests
- type checking
- CI

Si CI falla por los cambios realizados:

1. identificar la causa
2. corregirla
3. repetir validación local
4. hacer commit
5. hacer push
6. volver a verificar CI

## 7. Verification Agent

Cuando CI haya pasado, ejecutar un agente independiente de verificación.

Debe revisar:

- cumplimiento de la tarea
- cumplimiento de `docs/constitution.md`
- cumplimiento de `docs/specs.md`
- calidad del código
- tests
- regresiones
- arquitectura
- casos límite
- seguridad y manejo de datos

El hecho de que los tests pasen no implica automáticamente que la implementación sea correcta.

### Resultado

```text
Implementation
      ↓
Local validation
      ↓
Commit
      ↓
Push
      ↓
GitHub Actions
      ↓
Verification Agent
      │
      ├── PASS → Done
      │
      └── FAIL → Fix → Repeat
```

Una tarea no debe marcarse como terminada mientras el agente de verificación tenga problemas pendientes.
