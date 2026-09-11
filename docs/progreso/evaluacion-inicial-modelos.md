# Evaluación inicial del modelo

## Estado: Completada

## Contexto

Primera evaluación real del modelo sobre el subconjunto de test, ejecutada con la infraestructura de evaluación existente (esp §7). Hasta ahora solo se habían persistido las `val_loss` de entrenamiento; esta evaluación genera métricas por atributo sobre el test set.

## Modelo evaluado

| Campo | Valor |
|-------|-------|
| Arquitectura | ResNet18 (pretrained) + classifier head (Dropout 0.4 → 512 → ReLU → Dropout → 40) |
| Checkpoint | `checkpoints_40k/best_model.pt` (época 2, val_loss=0.208) |
| Dataset | CelebA subset 40K, 40 atributos binarios |
| Test set | 2,999 muestras (7.5% del subset, split idéntico al entrenamiento: 77.5/15/7.5, seed 42) |
| Preprocessing | Resize 224x224 + ToTensor (mismo que el entrenamiento) |

## Implementación

Se creó `scripts/evaluate.py`, que reproduce el split de `train_subset.py`, ejecuta inferencia en test y genera:

- `evaluation_results/metrics.json` — métricas globales
- `evaluation_results/attribute_summary.csv` — métricas por atributo (precision, recall, F1, support, tasas)
- `evaluation_results/error_analysis.json` — muestras de error
- `evaluation_results/thresholds.json` — thresholds optimizados por atributo (grid search F1)
- `evaluation_results/margins.json` — márgenes de incerteza estimados offline
- `evaluation_results/summary.json` — resumen con top-5 peores/mejores atributos

## Resultados globales

| Métrica | Threshold fijo (0.5) | Thresholds optimizados |
|---------|----------------------|-------------------------|
| Accuracy (exact match) | 0.025 | 0.015 |
| Precision (micro) | 0.839 | 0.738 |
| Recall (micro) | 0.731 | 0.841 |
| F1 (micro) | 0.781 | 0.786 |
| **Macro F1** | **0.661** | **0.720** |
| Hamming Loss | 0.091 | 0.102 |
| PR-AUC (micro) | 0.889 | — |
| ROC-AUC (macro) | 0.929 | — |

La optimización de thresholds mejora la Macro F1 en +0.059 puntos (39/40 atributos mejoran). Los scores del modelo están descalibrados: los thresholds óptimos se alejan de 0.5 (rango 0.17–0.64).

## Features donde más se equivoca

Top-12 peores atributos por F1 (threshold fijo):

| # | Atributo | F1 | Precision | Recall | Support | Tasa+ real | Tasa predichos |
|---|----------|------|-----------|--------|---------|-----------|----------------|
| 1 | Wearing_Necklace | 0.024 | 0.667 | 0.012 | 324 | 10.8% | 0.2% |
| 2 | Big_Lips | 0.305 | 0.735 | 0.192 | 650 | 21.7% | 5.7% |
| 3 | Rosy_Cheeks | 0.361 | 0.651 | 0.250 | 164 | 5.5% | 2.1% |
| 4 | Oval_Face | 0.371 | 0.640 | 0.261 | 811 | 27.0% | 11.0% |
| 5 | Mustache | 0.380 | 0.625 | 0.273 | 165 | 5.5% | 2.4% |
| 6 | Pointy_Nose | 0.391 | 0.589 | 0.293 | 772 | 25.7% | 12.8% |
| 7 | Narrow_Eyes | 0.414 | 0.633 | 0.307 | 348 | 11.6% | 5.6% |
| 8 | Double_Chin | 0.424 | 0.628 | 0.320 | 153 | 5.1% | 2.6% |
| 9 | Chubby | 0.445 | 0.711 | 0.324 | 182 | 6.1% | 2.8% |
| 10 | Blurry | 0.476 | 0.473 | 0.479 | 146 | 4.9% | 4.9% |
| 11 | Bags_Under_Eyes | 0.502 | 0.704 | 0.391 | 681 | 22.7% | 12.6% |
| 12 | Pale_Skin | 0.521 | 0.663 | 0.430 | 128 | 4.3% | 2.8% |

### Patrón de fallo dominante

Los peores atributos comparten el mismo síntoma: **`prediction_rate << positive_rate`** — el modelo predice "No" casi siempre, con recalls de 1–40%. Es un sesgo hacia la clase negativa.

Correlación `support→F1`: **r = 0.512** — cuanto más raro el atributo, peor F1. Causas raíz identificadas:

1. **`train_subset.py` no usa `pos_weight`**: `MultilabelLoss()` se instancia sin configuración (`pos_weight=None`), ignorando el balanceo de clases que el `Trainer` completo sí soporta. BCE sin pesos empuja los logits hacia abajo en clases minoritarias.
2. **Threshold fijo 0.5 subóptimo**: la optimización por atributo sube la Macro F1 de 0.661 a 0.720.
3. **Entrenamiento sin normalización ImageNet**: `train_subset.py` aplica solo `Resize + ToTensor`; el backbone preentrenado espera mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]. Dataset shift que desperdicia el conocimiento del transfer learning.

### Mejores features (contexto)

`Male` (F1=0.978), `No_Beard` (0.970), `Eyeglasses` (0.938), `Mouth_Slightly_Open` (0.928), `Young` (0.918). Son atributos frecuentes y visualmente distintivos.

## Alternativas de mejora propuestas

### P1 — Correcciones inmediatas (alto impacto, bajo esfuerzo)

| Alternativa | Detalle | Impacto esperado |
|-------------|---------|------------------|
| Optimizar thresholds por atributo | Escribir thresholds optimizados a `config/inference.yaml` para producción | Macro F1 0.66 → 0.72 |
| Entrenar con `pos_weight` automático | Activar balanceo de clases en `train_subset.py` y el flujo oficial | +recall en atributos con recall < 0.4 |
| Normalización ImageNet en training | Aplicar `Normalize` estándar en entrenamiento y evaluación de forma consistente | Mejora consistente de todo el modelo |

### P2 — Mejoras de datos (medio impacto)

| Alternativa | Detalle |
|-------------|---------|
| Usar las 24 features visualmente observables | `Oval_Face`, `Big_Lips`, `Pointy_Nose`, `Narrow_Eyes`, `Big_Nose`, `Attractive` son subjetivas con ruido de anotación y no son bien aprendidas |
| Face alignment con landmarks | Normalizar pose (ojos/boca) antes del crop; experimento controlado contra baseline (esp §5.5) |
| Augmentation más agresiva | `train_subset.py` no aplica la augmentation configurada en `training.yaml` (solo Resize+ToTensor) |

### P3 — Arquitectura / entrenamiento

| Alternativa | Detalle |
|-------------|---------|
| Focal Loss | Ataca directamente el under-recall de clases raras |
| Class-aware sampling | Sobremuestrear batches con clases minoritarias |
| Más épocas con early stopping en F1 | El mejor checkpoint fue época 2/10; el modelo aún no convergía |
| Calibración (Platt / isotónica) | Los scores están descalibrados; mejora confianza y márgenes |

### P4 — Arquitectura (mayor esfuerzo)

- Head multi-tarea por grupos de atributos (facial hair, hair style, accessories).
- ResNet34/50 o backbone moderno para subir el techo de ROC-AUC (0.93 actual).

## Archivos creados/modificados

```
scripts/evaluate.py                # Script de evaluación reutilizable
evaluation_results/                # Artefactos de la evaluación
├── metrics.json
├── attribute_summary.csv
├── error_analysis.json
├── thresholds.json
├── margins.json
└── summary.json
docs/progreso/evaluacion-inicial-modelos.md   # Este documento
```

## Verificación

- `uv run ruff check scripts/evaluate.py`: OK
- Inferencia en 2,999 muestras en ~144s (CPU)

## Próximos pasos

- Aplicar correcciones P1 (thresholds a config, pos_weight, normalización) y re-entrenar.
- Reescribir thresholds/márgenes estimados a `config/inference.yaml`.

---

*Referencia de especificación: `docs/specs.md` §7 (Evaluation).*