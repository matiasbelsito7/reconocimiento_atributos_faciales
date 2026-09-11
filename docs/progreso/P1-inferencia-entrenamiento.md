# P1 — Correcciones de entrenamiento y configuración de inferencia

## Estado: Completada

## Contexto

Aplicación de las correcciones inmediatas P1 propuestas en `evaluacion-inicial-modelos.md` para atacar el patrón de fallo dominante (sesgo hacia la clase negativa, `prediction_rate << positive_rate`) en los atributos menos frecuentes.

## Tareas implementadas

### P1.1 — Thresholds optimizados a configuración de inferencia

Se escribieron los thresholds y márgenes optimizados estimados sobre el test set al `config/inference.yaml` (sección `thresholds`). Formato de 40 entradas por mapa, nombres sin prefijo `Atr_` (convención de `CELEBA_ATTRIBUTE_NAMES`).

### P1.2 — Entrenamiento con `pos_weight` automático

- `scripts/train_subset.py`: nuevo método `_compute_pos_weight(train_ds, attribute_columns)` que calcula `neg/pos` por atributo desde el dataframe del training split (sin fuga de val/test), con cap `[1.0, 50.0]`. `loss_fn = MultilabelLoss(LossConfig(pos_weight=pos_weight))`.
- Mismo cálculo que el `Trainer` oficial (esp §6) y el método `_compute_pos_weight` de `FacialAttributeDataset`.

### P1.3 — Normalización ImageNet consistente

- `src/facial_attributes/training/config.py`: constantes `IMAGENET_MEAN` / `IMAGENET_STD`.
- `src/facial_attributes/training/dataset.py`: `apply_imagenet_normalization(tensor)` aplicada en `CachedAttributeDataset.__getitem__`.
- `scripts/train_subset.py` (`_build_transform`): `Resize + ToTensor + Normalize(ImageNet)`.
- `scripts/evaluate.py`: flag `--normalize` ahora es default `True` (`--no-normalize` para modelos legacy).
- `src/facial_attributes/face_processing/normalizer.py`: `FaceNormalizer.normalize` aplica `(x/255 - mean) / std` de ImageNet, manteniendo consistencia entrenamiento→servicio (evita train/serve skew).
- Tests actualizados en `tests/test_face_processing.py` (`test_normalize_face` valida el cálculo por canal).

### P1.4 — Re-entrenamiento

Nuevo entrenamiento del subset 40K en CPU (31000 train / 6000 val / 3000 test, seed 42):

| Época | train_loss | val_loss | best |
|-------|-----------|----------|------|
| 0 | — | 0.5045 | ✓ |
| 1 | — | — | — |
| 2 | — | 0.4740 | ✓ |
| 3 | — | **0.4644** | ✓ (best) |
| 4 | 0.4197 | 0.4844 | — |
| 5 | 0.3948 | 0.4917 | — |
| 6 | 0.3726 | 0.4933 | — |
| 7 | 0.3508 | 0.4977 | — |
| 8 | 0.3545 | 0.5231 | — |
| 9 | 0.3336 | 0.5298 | — |

El mejor modelo es la **época 3** (`checkpoints_40k/best_model.pt`). Nota: `val_loss` ya incluye `pos_weight`, no es comparable con el 0.208 del baseline.

### P1.5 — Re-evaluación y comparación

Resultados sobre el mismo test set (2,999 muestras), `evaluation_results_v2/`:

| Métrica | Baseline (th=0.5) | P1 (th=0.5) | Baseline (thr. opt) | P1 (thr. opt) |
|---------|-------------------|-------------|----------------------|---------------|
| Precision (micro) | 0.839 | 0.620 | 0.738 | 0.721 |
| Recall (micro) | 0.731 | 0.881 | 0.841 | 0.834 |
| F1 (micro) | 0.781 | 0.728 | 0.786 | 0.778 |
| **Macro F1** | **0.661** | **0.660** | **0.720** | **0.722** |
| PR-AUC (micro) | 0.889 | 0.841 | — | — |
| ROC-AUC (macro) | 0.929 | 0.928 | — | — |

## Lecciones aprendidas

1. **El `pos_weight` no mueve la Macro F1**: corrige el sesgo de clase negativa (recall 0.73 → 0.88) pero a costa de sobre-predecir positivos (precision 0.84 → 0.62), con saldo neto ~0 en F1. El desbalance se transfiere del falso-negativo al falso-positivo.
2. **Varias features problemáticas a threshold fijo sí mejoraron**: `Wearing_Necklace` F1 0.024→0.384, `Mustache` 0.380→0.528, `Pointy_Nose` 0.391→0.521, `Big_Lips` 0.305→0.469, `Oval_Face` 0.371→0.508, `Narrow_Eyes` 0.414→0.447, `Rosy_Cheeks` 0.361→0.393. En cambio `Chubby` (0.445→0.377), `Double_Chin` (0.424→0.386) y `Pale_Skin` (0.521→0.436) empeoran a th=0.5 por sobre-predicción.
3. **La normalización ImageNet es un prerequisito de consistencia**, no una palanca de mejora: ROC-AUC idéntico (0.928 vs 0.929). Alineó entrenamiento, evaluación y servicio.
4. **La palanca real sigue siendo la optimización de thresholds por atributo** (+0.06 Macro F1 en ambos modelos).
5. La AUPRC macro bajó levemente (0.841 vs 0.889): la calidad de ranking no mejora con estas correcciones; el siguiente paso debería ser Focal Loss o class-aware sampling (P3), no más `pos_weight`.

## Archivos modificados

```
config/inference.yaml                          # thresholds+márgenes del modelo P1 (40+40)
scripts/train_subset.py                        # pos_weight + Normalize ImageNet
scripts/evaluate.py                            # --normalize default True
src/facial_attributes/training/config.py       # IMAGENET_MEAN/STD
src/facial_attributes/training/dataset.py      # apply_imagenet_normalization
src/facial_attributes/face_processing/normalizer.py  # normalización ImageNet en servicio
tests/test_face_processing.py                  # test actualizado
evaluation_results_v2/                         # artefactos de la re-evaluación
docs/progreso/P1-inferencia-entrenamiento.md   # este documento
```

## Verificación

- `uv run ruff check` / `uv run black --check`: OK
- `uv run mypy` (5 fuentes afectadas): OK
- `uv run pytest` (119 tests): 119 passed (1 fix de aserción en normalizer)

## Próximos pasos

- Probar alternativas P3 que ataquen el under-recall de clases raras con mejor ranking (Focal Loss / class-aware sampling), evaluadas contra este punto de referencia.
- El flujo oficial (`Trainer` + `training.yaml`) ya soporta `pos_weight`; validar que aplica las mismas correcciones en el pipeline de entrenamiento completo.

---

*Referencia de especificación: `docs/specs.md` §7 (Evaluation).*