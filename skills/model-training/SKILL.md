# Model Training

## Purpose

Ejecutar experimentos de entrenamiento de modelos de reconocimiento de atributos faciales.

## When to use

- Al entrenar un nuevo modelo
- Al reentrenar con nuevos datos
- Al ajustar hiperparámetros
- Al comparar configuraciones

## Procedure

### 1. Pre-training

- Verificar que los datos estén validados (`skills/data-validation/SKILL.md`)
- Confirmar splits de entrenamiento/validación/prueba
- Configurar semilla para reproducibilidad
- Definir hiperparámetros del experimento

### 2. Configure

- Establecer semilla aleatoria
- Configurar hiperparámetros: learning rate, batch size, épocas
- Definir criterio de early stopping
- Seleccionar métrica de validación

### 3. Execute

```bash
uv run python -m src.training.train --config configs/experiment.yaml
```

Monitorear:
- Pérdida de entrenamiento y validación
- Métricas por época
- Tiempo de entrenamiento

### 4. Checkpoints

- Guardar checkpoints periódicamente
- Mantener el mejor modelo según métrica de validación
- Almacenar modelo con metadatos asociados

### 5. Post-training

- Registrar experimento en MLflow (`skills/experiment-tracking/SKILL.md`)
- Guardar configuración utilizada
- Documentar resultados obtenidos

## Restrictions

- No usar datos de prueba durante el entrenamiento
- Mantener reproducibilidad con semillas
- Versionar configuraciones y modelos
- No modificar datos raw

## Related

- `docs/specs.md` §6 (Training)
- `docs/constitution.md` §2 (ML Principles)
- `skills/experiment-tracking/SKILL.md`
