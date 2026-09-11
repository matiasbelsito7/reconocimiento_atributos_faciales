# Data

Datos del proyecto. Se mantiene la separación `raw`/`processed`; los datos crudos no se modifican manualmente.

## Estructura

- `raw/images/` - Imágenes del subset de CelebA (JPG).
- `raw/annotations/` - Anotaciones por atributo: CSV con `image_id` y columnas `Atr_*` (una por atributo facial).
- `processed/` - Datos derivados para entrenamiento: `celeba_subset_5000.csv` y `celeba_subset_40000.csv`.
- `DATASETS.md` - Documentación de detalle de los datasets.

## Módulo Python (`src/facial_attributes/data/`)

- `dataset.py` - Define la lista de los 24 atributos visualmente observables (`OBSERVABLE_ATTRIBUTE_NAMES`) y la clase `DatasetManager`: carga de anotaciones, detección de columnas de atributos (`Atr_*`), filtrado de observables (case-insensitive, compatible con nombres reales de CelebA), división train/val/test con seed reproducible, guardado de splits e información resumida del dataset.
- `validation.py` - Clase `DataValidator`: valida el formato del CSV de anotaciones, existencia y legibilidad de imágenes (muestreo de 100), duplicados y distribución de clases para detectar desbalance. Retorna un reporte `PASS`/`FAIL` con `errors`, `warnings` y estadísticas.
- `__init__.py` - Reexporta `DatasetManager` y `DataValidator` para consumo externo.
