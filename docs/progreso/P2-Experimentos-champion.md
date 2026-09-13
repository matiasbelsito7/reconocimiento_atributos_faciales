# P2 - Campaña de experimentos y decisiones de champion

## Estado: Completada

## Contexto

Campaña P2 para mejorar el modelo inicial P1 (40 atributos, entrenamiento sin augmentation). Se ejecutaron tres corridas comparables sobre el mismo dataset (`celeba_subset_40000`, split 31000/6000/3000, seed 42, CPU, ~1 h/época):

1. **Baseline no-aug**: 10 épocas, 40 atributos. Corrió en silencio **sin augmentation** por un bug de config (ver decisiones).
2. **Exp A**: 10 épocas, 40 atributos, augmentation activa → primer champion (`v2.0.0`).
3. **Exp B**: 10 épocas, **24 atributos observables**, augmentation activa → champion actual (`v2.1.0`).

Todas con resnet18, lr 3e-4, batch 32, dropout 0.4, BCE-with-logits con `pos_weight`, transform Resize(224)+Normalize ImageNet.

## Resultados

### Validación (mejor val_loss por corrida)

| Corrida | Atributos | Mejor val_loss | Época | Mejor val_f1 |
|---|---|---|---|---|
| Baseline no-aug | 40 | 0.4644 | 4 | 0.764 (ép. 10) |
| Exp A | 40 | 0.4560 | 7 | 0.743 (ép. 10) |
| Exp B | 24 | 0.3683 | 8 | 0.779 (ép. 10) |

### Test — Exp A vs no-aug (40 atributos, threshold 0.5)

| Métrica | Exp A | Baseline no-aug | Ganador |
|---|---|---|---|
| Precision (micro) | 0.6233 | 0.6202 | Exp A |
| Recall (micro) | 0.8866 | 0.8814 | Exp A |
| F1 (micro) | 0.7320 | 0.7281 | Exp A |
| Macro F1 | 0.6620 | 0.6597 | Exp A |
| Hamming Loss | 0.1444 | 0.1464 | Exp A |
| Avg Precision | 0.8480 | 0.8409 | Exp A |
| ROC-AUC | 0.9302 | 0.9278 | Exp A |

**Decisión**: Exp A gana 6/6 métricas y 23/40 atributos → promovido como champion.

### Test — Exp B vs Exp A (24 columnas observables, threshold 0.5)

Para comparación justa, Exp A (40 outputs) se re-evaluó recortando predicciones a las 24 columnas observables sobre el mismo test split.

| Métrica | Exp B | Exp A (obs24) | Ganador |
|---|---|---|---|
| Precision (micro) | 0.6589 | 0.6584 | Exp B |
| Recall (micro) | 0.9315 | 0.9174 | Exp B |
| F1 (micro) | 0.7718 | 0.7666 | Exp B |
| Macro F1 | 0.7074 | 0.6993 | Exp B |
| Hamming Loss | 0.1099 | 0.1115 | Exp B |
| Avg Precision | 0.9014 | 0.8866 | Exp B |
| ROC-AUC | 0.9614 | 0.9584 | Exp B |

**Decisión**: Exp B gana 6/6 métricas y 15/24 atributos por F1 (top: `5_o_Clock_Shadow` +0.071, `Bald` +0.062, `Wearing_Necktie` +0.062) → promovido como champion actual.

## Estado del Model Registry

| Modelo | ID | Versión | Estado |
|---|---|---|---|
| Exp A (baseline no-aug no registrado) | - | - | - |
| Exp A | `18058005-1571-4278-b593-8edc6cd5d16c` | 2.0.0 | archived |
| **Exp B** | `2e095b29-3a87-4f70-bef0-a79823d1e1be` | **2.1.0** | **production** |

La promoción se realizó con `promote_candidate_to_production()` + `MajorityPromotionPolicy` (ver `docs/progreso/champion-promotion-mayoria.md`). Baseline no-aug quedó sin registrar por ser superado por Exp A en todas las métricas; se conserva como artefacto en disco (`checkpoints_40k/best_model_noaug.pt`).

## Decisiones técnicas y aprendizajes

1. **Bug de config ocultaba la augmentation**: `LoggingConfig` rechazaba los keys `log_interval`/`save_tensorboard`/`tensorboard_dir` de `training.yaml`, por lo que `load_training()` fallaba y `train_subset.py` deshabilitaba la augmentation con un AVISO. Se corrigió el esquema (commit `1930eba`). El baseline "no-aug" es en realidad el resultado del bug, no una decisión deliberada.
2. **La augmentation sí aporta**: Exp A > baseline en casi todo (especialmente atributos de bajo soporte: `Wearing_Hat` +0.154). Pero perjudicó algunos raros (`Bald` -0.103, `Pale_Skin` -0.100, `Blurry` -0.070).
3. **Restringir a atributos observables mejora**: Exp B supera a Exp A reutilizando la misma capacidad en 24 columnas verificables. El alcance del champion pasa de 40 a 24 columnas.
4. **Metodología de comparación**: se decide con el reporte de **threshold fijo 0.5** (donde las métricas de ranking son válidas). El reporte "optimizado" calcula ranking metrics sobre predicciones binarizadas (AvgPrec/ROC artificialmente bajos) y usa thresholds sobreajustados al test, por lo que **no** se usa para decidir champion.
5. **Propagación de `attribute_columns` en cache**: la rama de cache `.npy` de `train_subset.py` ignoraba el subset; corregido (commit `264dd5b`) pasando `attribute_columns` a `CachedAttributeDataset` y agregando CLI (`--attribute-subset observable`, `--checkpoint-dir`, `--cache-dir`).
6. **Campo `best_val_loss` del checkpoint es stale**: `CheckpointManager.save_checkpoint` guarda el mejor *anterior* (se actualiza después). Cosmético: el peso de `best_model.pt` es correcto; solo el valor mostrado es el del best previo.
7. **No existe cache `.npy`**: Exp A y Exp B re-redimensionaron imágenes por época (~1 h/época en CPU). Opción futura: generar `data/processed/cache_40000/npy` para acelerar.

## Artefactos

- `checkpoints_40k/best_model.pt` → Exp A (época 7). Copias: `best_model_noaug.pt`, `best_model_P1_ep3.pt`.
- `checkpoints_40k_obs24/best_model.pt` → Exp B (época 8).
- `evaluation_results_v2/exp_a/`, `.../no_aug/` → evaluaciones 40 atributos.
- `evaluation_results_v2/obs24/exp_b/`, `.../obs24/exp_a/` → evaluaciones 24 observables.
- `models/registry/registry.json` → registro con ambos candidatos y el champion.

## Autorreferencia

- `docs/specs.md` §13 (Model Registry), §13.6 (CD del modelo champion).
- `docs/constitution.md` §2 (ML Principles).
- `docs/progreso/champion-promotion-mayoria.md`, `docs/progreso/P2-expA-champion.md`.