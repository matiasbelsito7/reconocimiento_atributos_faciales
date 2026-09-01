# Inference Pipeline

## Purpose

Ejecutar inferencia de atributos faciales en nuevas imágenes.

## When to use

- Para predecir atributos en imágenes nuevas
- Para procesar imágenes en lote
- Para integración con aplicaciones

## Pipeline

```text
Input Image → Preprocessing → Face Detection → Model Inference → JSON Output
```

## Procedure

### 1. Load model

- Cargar modelo entrenado y configuración
- Verificar compatibilidad de versión

### 2. Preprocess

- Aplicar transformaciones estándar
- Normalizar imagen

### 3. Detect faces

- Detectar bounding boxes de rostros
- Extraer y normalizar cada rostro
- Manejar errores: sin rostro, múltiples rostros

### 4. Predict

- Ejecutar inferencia por cada rostro
- Obtener scores por atributo
- Aplicar threshold para predicciones binarias

### 5. Format output

```json
{
  "faces": [
    {
      "bbox": [x, y, w, h],
      "attributes": {
        "smiling": 0.92,
        "glasses": 0.15
      }
    }
  ]
}
```

## Error handling

- **Sin rostro detectado**: reportar error, no fallar silenciosamente
- **Baja calidad**: decidir comportamiento (advertencia o skip)
- **Modelo no disponible**: reportar error claro

## Restrictions

- No almacenar imágenes innecesariamente
- Mantener trazabilidad de predicciones
- Versionar modelo utilizado

## Related

- `docs/specs.md` §9 (Inference)
- `skills/model-training/SKILL.md`
- `skills/model-evaluation/SKILL.md`
