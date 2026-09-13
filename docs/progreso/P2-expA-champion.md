# P2 - Exp A: Champion registrado

## Estado: Completada

## Contexto

Dentro del pipeline P2 se entrenó el experimento **Exp A**: 10 épocas sobre los 40 atributos con **augmentation config-driven** activada (`flip_h`, rotación 15°, brillo y contraste en `[0.8, 1.2]`). El entrenamiento completo se registró tras corregir el bug de `LoggingConfig` (commit `1930eba`) que deshabilitaba la augmentation en silencio.

El mejor modelo de Exp A (época 7, val_loss 0.4560) se evaluó contra la corrida sin augmentation en el mismo test split (2999 muestras, seed 42). Con la política `MajorityPromotionPolicy`, Exp A gana **6/6 métricas globales** y mejora el F1 en **23 de 40 atributos**, por lo que se registra como primer champion del Model Registry.

## Resultados (test set, threshold fijo 0.5)

| Métrica | Exp A (época 7) | No-aug | Resultado |
|---|---|---|---|
| Precision (micro) | 0.6233 | 0.6202 | Exp A |
| Recall (micro) | 0.8866 | 0.8814 | Exp A |
| F1 (micro) | 0.7320 | 0.7281 | Exp A |
| Macro F1 | 0.6620 | 0.6597 | Exp A |
| Hamming Loss | 0.1444 | 0.1464 | Exp A |
| Avg Precision | 0.8480 | 0.8409 | Exp A |
| Macro ROC-AUC | 0.9302 | 0.9278 | Exp A |

- **Policy**: `Candidate wins 6/6 metrics, promotes as new champion.`
- **F1 por atributo**: 23/40 mejores en Exp A (top ganancia: `Wearing_Hat` +0.154, `Rosy_Cheeks` +0.081, `Chubby` +0.061).

## Cambios realizados

- **Registro del champion** (`models/registry/registry.json`): modelo `facial_attribute_classifier v2.0.0`, estado `production`, model_id `18058005-1571-4278-b593-8edc6cd5d16c`.
  - Métricas globales + F1 per-attribute (40) del reporte de test.
  - Config real: resnet18, 40 atributos, lr 3e-4, batch 32, 10 épocas, dropout 0.4, `bce_with_logits`, augmentation activa.
  - Dataset: `celeba_subset_40000` (train 0.775 / val 0.15 / test 0.075, seed 42).
  - Artefactos: `checkpoints_40k/best_model.pt` + reportes `evaluation_results_v2/exp_a/`.
  - Tags: `experiment=exp_a`, `augmentation=true`, `best_epoch=7`.
- **`.gitignore`**: excepción para que el metadata del Model Registry (`models/registry/registry.json`) sea versionable, manteniendo ignorados los pesos (`models/*`).
- **Reportes de evaluación** commiteados: `evaluation_results_v2/exp_a/` (champion) y `evaluation_results_v2/no_aug/` (corrida base de comparación).

## Decisiones técnicas

- La promoción se hizo con `promote_candidate_to_production()` del `ModelRegistry` JSON (backend real del champion; MLflow local aún sin runs).
- Exp A ya era el contenido de `checkpoints_40k/best_model.pt`; el registro de champion solo captura metadata y provenance, sin tocar el artefacto en disco.
- Métricas de decisión: evaluación con threshold 0.5 (donde las métricas de ranking son válidas). Las métricas "optimizadas" con thresholds tuneados sobre test no se usan para la decisión.

## Referencia

- `docs/specs.md` §13 (Model Registry) y §13.6 (CD del modelo champion).
- `docs/constitution.md` §2 (ML Principles).
- Feature de promoción por mayoría: `docs/progreso/champion-promotion-mayoria.md`.