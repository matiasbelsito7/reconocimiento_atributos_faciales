# Data Validation

## Purpose

Validar la integridad y formato de datasets de imágenes y anotaciones antes de procesarlos.

## When to use

- Antes de agregar un nuevo dataset al proyecto
- Después de modificar anotaciones
- Antes de ejecutar pipelines de preprocessing
- Como parte del proceso de calidad de datos

## Checks

### 1. Integridad de imágenes

- Verificar que cada archivo de imagen sea legible
- Detectar imágenes corruptas o con formato incorrecto
- Reportar imágenes que no puedan abrirse

### 2. Formato de anotaciones

- Verificar que el CSV tenga las columnas esperadas
- Validar que los valores binarios sean 0 o 1
- Detectar valores faltantes o inválidos

### 3. Consistencia

- Verificar que toda imagen tenga una anotación asociada
- Detectar anotaciones sin imagen correspondiente
- Identificar imágenes duplicadas

### 4. Distribución

- Reportar distribución de cada atributo
- Identificar atributos con muy pocos ejemplos
- Detectar desbalance significativo

## Procedure

```bash
# Ejecutar validación (ejemplo)
uv run python -m src.data.validate --data-dir data/raw
```

El script debe generar un reporte con:
- Total de imágenes verificadas
- Imágenes corruptas o ilegibles
- Anotaciones inválidas
- Duplicados detectados
- Distribución de atributos

## Output

Reporte de validación que indique:
- PASS: dataset válido
- WARN: advertencias que no impiden el uso
- FAIL: errores que deben corregirse antes de continuar

## Related

- `docs/specs.md` §2.5 (Validación de datos)
- `skills/dataset-management/SKILL.md` (gestión del dataset)
