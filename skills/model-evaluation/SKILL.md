# Model Evaluation

## Purpose

Evaluar el rendimiento de modelos entrenados de reconocimiento de atributos faciales.

## When to use

- Después de entrenar un modelo
- Al comparar modelos
- Para análisis de errores
- Para validación final

## Metrics

### Per attribute

- Exactitud (accuracy)
- Precisión (precision)
- Recall
- F1-score

### Global

- Exactitud global
- Micro/macro/macro promediadas

## Procedure

### 1. Load model

- Cargar modelo entrenado
- Cargar conjunto de prueba (nunca usado en entrenamiento)

### 2. Run inference

- Ejecutar predicciones sobre conjunto de prueba
- Almacenar scores y predicciones binarias

### 3. Calculate metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

Calcular métricas por atributo y globalmente.

### 4. Analyze errors

- Identificar atributos con peor rendimiento
- Analizar patrones en predicciones incorrectas
- Visualizar ejemplos de errores

### 5. Generate report

- Tabla de métricas por atributo
- Métricas globales
- Análisis cualitativo de errores
- Comparación con modelos anteriores (si existe)

## Output

- Reporte de evaluación con métricas
- Análisis de errores
- Recomendaciones para mejora

## Related

- `docs/specs.md` §7 (Evaluation)
- `docs/constitution.md` §2 (ML Principles)
- `skills/model-training/SKILL.md`
