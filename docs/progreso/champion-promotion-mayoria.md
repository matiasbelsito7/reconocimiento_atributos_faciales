# Champion promotion por mayoría de métricas

## Estado: Completada

## Contexto

Al correr experimentos (P2: Exp A augmentation vs no-aug), se detectó que la promoción del modelo *champion* podía decidirse por una única métrica (F1), promoviendo un modelo peor en el resto de los aspectos. Se requiere que el champion solo se reemplace cuando el candidato efectivamente mejora en la **mayoría** de las métricas de evaluación.

## Cambios implementados

### `src/facial_attributes/retraining/criteria.py` — `MajorityPromotionPolicy`

Nueva política `MajorityPromotionPolicy.evaluate(candidate, champion) -> PromotionDecision`:

- Compara las métricas de `ModelMetrics` con dirección conocida: `accuracy`, `precision`, `recall`, `f1_score`, `average_precision` (mayor = mejor) y `hamming_loss` (menor = mejor).
- Cuenta `wins` / `losses` / `ties` por métrica.
- **Denominador**: solo métricas "activas" (al menos un modelo con valor != 0), evitando que métricas no evaluadas (defaults 0.0) diluyan la decisión.
- **Regla de mayoría**: `should_promote = wins > active / 2` — se requiere ganar en **más de la mitad** de las métricas comparadas, no basta una sola.
- `PromotionDecision` expone wins/losses/ties, detalle por métrica y summary legible.

### `src/facial_attributes/model_registry/registry.py`

- `promote_candidate_to_production(model_id, policy=None) -> PromotionDecision`:
  - Sin champion vigente → promociona directamente.
  - Con champion → compara contra el vigente; **solo reemplaza** (archivando el anterior) si la política de mayoría se cumple.
- `compare_models`: el `winner` ahora se decide por la política de mayoría (antes solo por `f1_score`).

## Tests

`tests/test_retraining.py` (`TestMajorityPromotionPolicy`):

- Promueve cuando el candidato gana todas las métricas.
- **No** promueve cuando mejora una sola métrica y pierde el resto.
- `hamming_loss` menor cuenta como mejora.
- Empate (1 win, resto ties) no promueve.

`tests/test_model_registry.py`:

- Promueve candidato sin champion.
- Promueve candidato que gana la mayoría (archiva el champion previo).
- Mantiene champion cuando el candidato no gana la mayoría.

## Verificación

- `uv run pytest` (275 tests): 275 passed.
- `uv run ruff check .` / `uv run black`: OK.
- `uv run mypy` (criteria.py + registry.py): OK.

## Autorreferencia

- `docs/specs.md` §13 (Model Registry), §13.6 (CD del modelo champion).