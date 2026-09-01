# Code Quality

## Purpose

Verificar la calidad del código antes de commits y durante desarrollo.

## When to use

- Antes de cada commit
- Después de modificar código
- Como parte del workflow de desarrollo
- Para mantener consistencia en el proyecto

## Tools

### Ruff (linting)

```bash
uv run ruff check .
```

Detecta: errores de sintaxis, imports no utilizados, estilo de código.

### Black (formatting)

```bash
uv run black .
```

Formatea el código de forma consistente.

### mypy (type checking)

```bash
uv run mypy .
```

Verifica type hints y consistencia de tipos.

### pytest (tests)

```bash
uv run pytest
```

Ejecuta la suite de tests.

### pre-commit (all-in-one)

```bash
uv run pre-commit run --all-files
```

Ejecuta todas las verificaciones de calidad.

## Procedure

### 1. Run all checks

```bash
uv run pre-commit run --all-files
```

### 2. Fix issues

- Si Ruff reporta errores: corregir o justificar
- Si Black modifica archivos: revisar cambios
- Si mypy reporta errores: corregir tipos
- Si pytest falla: corregir tests o código

### 3. Re-run

Repetir hasta que todas las herramientas pasen sin errores.

### 4. Verify

```bash
git status
git diff
```

Verificar que no haya cambios inesperados.

## Restrictions

- No ocultar errores para hacer pasar herramientas
- No eliminar tests sin justificación
- No deshabilitar checks de calidad
- Mantener type hints en código Python

## Related

- `docs/constitution.md` §3 (Calidad de software)
- `skills/development-workflow/SKILL.md`
