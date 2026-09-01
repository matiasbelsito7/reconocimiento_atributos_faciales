# Facial Attribute Recognition

Reconocimiento multilabel de atributos faciales visualmente observables mediante Machine Learning y Deep Learning.

## Descripción

Este sistema predice la presencia o ausencia de múltiples atributos faciales simultáneamente, como:

- Presencia de gafas / lentes
- Sonrisa
- Barba
- Bigote
- Accesorios (pendientes, sombrero, etc.)
- Expresión facial (neutral, feliz, triste, etc.)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/matiasbelsito7/reconocimiento_atributos_faciales.git
cd reconocimiento_atributos_faciales

# Instalar dependencias
uv sync --all-groups
```

## Uso

```bash
# Ejecutar tests
make test

# Verificar código
make validate

# Formatear código
make format
```

## Stack

- Python 3.11+
- PyTorch
- torchvision
- scikit-learn
- OpenCV / Pillow
- MLflow
- pytest
- Ruff / Black / mypy

## Documentación

- [Constitución del proyecto](docs/constitution.md)
- [Especificaciones](docs/specs.md)
- [Tareas](docs/tasks.md)

## Licencia

MIT
