# Dataset Management

## Purpose

Gestionar el ciclo de vida de los datasets: adición, documentación y separación entre datos raw y processed.

## When to use

- Al agregar un nuevo dataset al proyecto
- Al documentar metadatos de un dataset
- Al separar datos raw de processed
- Al actualizar la documentación de datasets

## Operations

### 1. Add dataset

Pasos para agregar un dataset:

1. Almacenar imágenes en `data/raw/images/`
2. Crear CSV de anotaciones en `data/raw/annotations/`
3. Ejecutar validación (`skills/data-validation/SKILL.md`)
4. Documentar metadatos (ver abajo)

### 2. Document dataset

Registrar en `docs/datasets/` o en el README correspondiente:

- **Nombre**: identificador del dataset
- **Fuente**: origen del dataset
- **Licencia**: tipo de licencia
- **Fecha de obtención**: cuando se descargó/adquirió
- **Tamaño**: número de imágenes
- **Atributos**: lista de atributos incluidos
- **Formato**: estructura de anotaciones

### 3. Separate raw/processed

- `data/raw/`: datos originales, nunca modificar
- `data/processed/`: resultado de pipelines de transformación
- Cada transformación debe ser reproducible y versionada

## Restrictions

- Nunca modificar archivos en `data/raw/`
- Mantener trazabilidad de transformaciones
- Documentar toda operación sobre datasets
- Respetar licencias de los datasets

## Related

- `docs/specs.md` §2 (Data)
- `docs/constitution.md` §4 (Datos)
- `skills/data-validation/SKILL.md`
