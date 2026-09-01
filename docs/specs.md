# Specifications

Especificaciones funcionales y técnicas del sistema **Facial Attribute Recognition**.

Este documento describe el estado deseado del sistema: qué debe hacer, qué entradas recibe, qué salidas produce y qué restricciones debe cumplir.

---

## 1. System Overview

El sistema es un pipeline de reconocimiento multilabel de atributos faciales visualmente observables. Toma imágenes que contienen rostros y predice la presencia o ausencia de múltiples atributos faciales simultáneamente.

### Arquitectura general

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Configuration Module                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Management                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Dataset A │  │ Dataset B │  │   ...    │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Validation                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│    Training Pipeline      │   │   Inference Pipeline       │
│  ┌─────────────────────┐  │   │  ┌─────────────────────┐  │
│  │ Preprocessing (train)│  │   │  │ Preprocessing (inf) │  │
│  └─────────────────────┘  │   │  └─────────────────────┘  │
│  ┌─────────────────────┐  │   │  ┌─────────────────────┐  │
│  │   Face Processing    │  │   │  │   Face Processing    │  │
│  └─────────────────────┘  │   │  └─────────────────────┘  │
│  ┌─────────────────────┐  │   │  ┌─────────────────────┐  │
│  │      Training        │  │   │  │     Inference        │  │
│  └─────────────────────┘  │   │  └─────────────────────┘  │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│      Evaluation           │   │      Monitoring            │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Model Registry                              │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline de entrenamiento

```text
data → data_validation → preprocessing_train → face processing → training → evaluation
```

### Pipeline de inferencia

```text
input → preprocessing_inference → face processing → inference → monitoring
```

### Pipeline de reentrenamiento

```text
new_data → validation → merge_with_existing → retraining → evaluation → model_registry
```

### Componentes

- **Configuration**: Configuración centralizada del sistema (pipeline, modelos, entrenamiento, inferencia).
- **Data Management**: Gestión de múltiples datasets con soporte para diferentes fuentes y formatos.
- **Data Validation**: Validación de integridad y formato de datos antes del procesamiento.
- **Preprocessing**: Preparación de imágenes con pipelines separados para training e inference.
- **Face Processing**: Detección, extracción y normalización de rostros.
- **Training**: Entrenamiento del modelo de reconocimiento de atributos.
- **Evaluation**: Medición y análisis independiente del rendimiento del modelo.
- **Inference**: Predicción de atributos para nuevas imágenes.
- **Monitoring**: Seguimiento de rendimiento y drift en producción.
- **Model Registry**: Almacenamiento y versionado de modelos entrenados.
- **Application**: Interfaz o servicio que utiliza el modelo entrenado.

### Problema

Reconocimiento multilabel: cada rostro puede tener múltiples atributos activos simultáneamente. Los atributos son exclusivamente **visualmente observables** (expresiones, accesorios, características faciales visibles).

---

## 2. Data

### 2.1 Gestión de múltiples datasets

El sistema debe soportar múltiples datasets con diferentes características:

```text
datasets/
├── dataset_a/
│   ├── raw/
│   └── processed/
├── dataset_b/
│   ├── raw/
│   └── processed/
└── merged/
```

Cada dataset debe:
- Tener su propio directorio raw y processed
- Documentar fuente, licencia y formato de anotaciones
- Poder combinarse para entrenamiento

### 2.2 Entradas esperadas

- Imágenes que contengan al menos un rostro humano visible.
- Formatos soportados: JPEG, PNG.
- Cada imagen debe tener una anotación asociada que indique los atributos faciales presentes.

### 2.3 Estructura de datos

#### Imágenes

- Almacenamiento en directorio separado de anotaciones.
- Resolución mínima recomendada: suficiente para que el rostro sea detectable (no se fija un mínimo absoluto).
- Nombre de archivo identificador único.

#### Anotaciones

- Formato tabular (CSV o equivalente) con:
  - Identificador de la imagen.
  - Columnas binarias (0/1) para cada atributo facial soportado.
- Cada fila representa una imagen y sus atributos.
- Los atributos deben ser fijos y definidos antes del entrenamiento.

### 2.4 Atributos soportados

Los atributos deben ser **visualmente observables** en el rostro. Ejemplos válidos:

- Presencia de gafas / lentes
- Sonrisa
- Barba
- Bigote
- Accesorios (pendientes, sombrero, etc.)
- Expresión facial (neutral, feliz, triste, etc.)

**No se incluyen** atributos sensibles (ver Sección 5).

### 2.5 Separación de datos

- **Raw data**: Imágenes y anotaciones originales. Nunca se modifican directamente.
- **Processed data**: Resultado de pipelines de transformación. Generados reproduciblemente.
- **Merged data**: Combinación de datasets para entrenamiento.

### 2.6 Validación de datos

- Verificar integridad de archivos de imagen.
- Verificar que toda imagen tenga anotación asociada.
- Verificar que las anotaciones tengan el formato correcto.
- Detectar y reportar imágenes duplicadas.
- Reportar distribución de atributos para identificar desbalance.
- **Ejecutar antes de cualquier procesamiento** (etapa obligatoria).

### 2.7 Reproducibilidad y trazabilidad

- Cada transformación de datos debe ser versionada y ejecutable de forma idéntica.
- Registrar la procedencia de los datasets (fuente, licencia, fecha de obtención).
- Mantener un registro de transformaciones aplicadas.
- Versionar datos procesados para trazabilidad.

### 2.8 Versionado de datos

- Cada dataset procesado debe tener un identificador de versión.
- Los cambios en datos procesados deben quedar registrados.
- Permitir reversiones a versiones anteriores cuando sea necesario.

---

## 3. Preprocessing

El preprocessing prepara las imágenes para el face processing. Existen dos pipelines separados con necesidades diferentes.

### 3.1 Preprocessing para entrenamiento

Pipeline optimizado para batch processing y augmentation:

- Redimensionamiento a dimensiones estándar.
- Normalización de color (conversión a formato uniforme).
- Corrección básica de calidad.
- **Data augmentation** (optional): rotación, flips, cambios de brillo.
- Procesamiento en lote (batch).

### 3.2 Preprocessing para inferencia

Pipeline optimizado para latencia baja:

- Redimensionamiento a dimensiones estándar.
- Normalización de color.
- Sin augmentation.
- Procesamiento de imagen individual.
- Tiempo de respuesta mínimo.

### 3.3 Transformaciones compartidas

Ambos pipelines comparten:
- Normalización de píxeles consistente.
- Formato de salida uniforme.
- Registro de transformaciones aplicadas.

### 3.4 Restricciones

- Las transformaciones deben ser reproducibles.
- No se debe degradar innecesariamente la calidad de las imágenes.
- Las transformaciones aplicadas deben quedar registradas.
- Los pipelines de training e inference deben ser independientes.

### 3.5 Salida

- Imágenes procesadas listas para face processing.
- Mantener referencias a las anotaciones originales (training).
- Mantener metadata de la imagen original (inference).

---

## 4. Face Processing

El face processing es responsable de localizar y preparar los rostros para la inferencia de atributos.

### 4.1 Detección de rostros

- Detectar la presencia de al menos un rostro por imagen.
- Localizar la región del rostro (bounding box).
- Manejar imágenes con múltiples rostros: procesar cada rostro individualmente.

### 4.2 Extracción

- Recortar la región del rostro detectado.
- Mantener relación de aspecto cuando sea apropiado.

### 4.3 Normalización

- Redimensionar rostros extraídos a un tamaño uniforme para el modelo.
- Aplicar normalización de píxeles consistente con lo esperado por el modelo.

### 4.4 Manejo de errores

- Reportar imágenes donde no se detecte ningún rostro.
- Decidir comportamiento para rostros de baja calidad (parcialmente obstruidos, borrosos).

### 4.5 Salida

- Rostros extraídos y normalizados listos para inferencia.
- Metadatos de cada rostro: posición, tamaño, nivel de confianza de detección.

---

## 5. Model

### 5.1 Problema

- Clasificación multilabel: cada rostro tiene múltiples atributos binarios independientes.
- Cada atributo se predice como presente (1) o ausente (0).

### 5.2 Entrada

- Rostro normalizado (imagen de dimensiones fijas).
- Formato: tensor o arreglo de imagen estandarizado.
- **Opcional**: landmarks faciales, bounding box, metadata adicional.

### 5.3 Salida

- Vector de scores o probabilidades para cada atributo.
- Un score por cada atributo en el conjunto definido.
- Los scores deben representar la confianza de predicción (rango [0, 1] o equivalente).

### 5.4 Comportamiento esperado

- Predicción independiente por atributo (sin dependencias entre atributos en la salida).
- Capacidad de umbralizar scores para obtener predicciones binarias.
- Soporte para entrada extendida (landmarks, metadata) cuando esté disponible.

### 5.5 Decisiones pendientes

- **Arquitectura concreta del modelo** → pendiente de definición.
- **Función de pérdida** → pendiente de definición.
- **Estrategia de balanceo de clases** → pendiente de definición.
- **Uso de landmarks faciales** → pendiente de definición.

### 5.6 Restricciones

- No utilizar atributos sensibles como entrada o salida del modelo.
- El modelo debe poder ejecutarse en inferencia sin acceso a datos de entrenamiento.
- El modelo debe ser compatible con el pipeline de inference optimizado.

---

## 6. Training

### 6.1 Configuración

- Configuración reproducible: semillas, hiperparámetros, versión de dependencias.
- Registro de configuración antes de cada experimento.

### 6.2 Datasets

- Utilizar datasets con anotaciones de atributos faciales visualmente observables.
- Documentar fuente, licencia y tamaño de cada dataset utilizado.
- Separar estrictamente entrenamiento, validación y prueba.

### 6.3 Separación de datos

- Conjunto de entrenamiento: para ajustar pesos del modelo.
- Conjunto de validación: para selección de hiperparámetros y early stopping.
- Conjunto de prueba: para evaluación final independiente.
- No debe haber fuga de datos entre conjuntos.

### 6.4 Tracking de experimentos

- Registrar cada experimento con MLflow.
- Información a registrar: configuración, métricas, artefactos, duración.
- Comparar experimentos entre sí.

### 6.5 Checkpoints

- Guardar checkpoints durante el entrenamiento.
- Permitir reanudar entrenamiento desde un checkpoint.
- Mantener el mejor modelo según métrica de validación.

### 6.6 Métricas de entrenamiento

- Registrar métricas de entrenamiento y validación por época.
- Identificar sobreajuste (overfitting) y subajuste (underfitting).

### 6.7 Manejo de modelos

- Almacenar modelos entrenados con metadatos associados.
- Versionar modelos junto con su configuración.

---

## 7. Evaluation

La evaluación es un proceso independiente del entrenamiento que puede ejecutarse sin reentrenar el modelo.

### 7.1 Métricas relevantes

- Métricas multilabel estándar (pendiente de definir cuáles específicamente):
  - Exactitud por atributo
  - Precisión, recall, F1 por atributo
  - Métricas globales multilabel
- **No se han fijado thresholds ni métricas específicas aún** → pendiente de definición.

### 7.2 Evaluación por atributo

- Reportar rendimiento de cada atributo individualmente.
- Identificar atributos con mejor y peor rendimiento.
- Análisis de distribución de predicciones por atributo.

### 7.3 Análisis de errores

- Analizar patrones en predicciones incorrectas.
- Identificar atributos que el modelo confunde con frecuencia.
- Visualizar ejemplos de errores para comprensión cualitativa.
- Generar reporte de análisis de errores.

### 7.4 Calibración

- Evaluar si las probabilidades predichas reflejan frecuencias reales.
- **Pendiente de definir** si se requiere calibración explícita.

### 7.5 Robustez

- Evaluar rendimiento ante variaciones (iluminación, pose, oclusión parcial).
- **Pendiente de definir** requisitos específicos de robustez.

### 7.6 Criterios de comparación

- Comparar modelos usando las mismas métricas y conjunto de prueba.
- Documentar condiciones de comparación para que sea justa.
- Mantener historial de evaluaciones para tendencias.

### 7.7 Separación de evaluación

- La evaluación debe ejecutarse con el conjunto de prueba (nunca visto en entrenamiento).
- No debe haber dependencia con el pipeline de training.
- Los resultados deben registrarse en el Model Registry.

---

## 8. Explainability

### 8.1 Requisitos

- Capacidad de identificar qué regiones de la imagen influyeron en la predicción de cada atributo.
- Métodos de explicación: **pendiente de definir** (Grad-CAM, attention maps, etc.).

### 8.2 Formato de explicaciones

- Visualizaciones que muestre regiones de interés.
- Reportes que indiquen contribución relativa de atributos.

### 8.3 Restricciones

- Las explicaciones no deben requerir acceso a datos de entrenamiento.
- Deben ser interpretables por un usuario técnico.

---

## 9. Inference

### 9.1 Entrada

- Imagen (o imagen con rostro pre-extraído).
- Configuración de inferencia (thresholds, modo de operación).

### 9.2 Procesamiento

1. Preprocessing de la imagen.
2. Detección y extracción de rostros.
3. Normalización del rostro.
4. Predicción de scores para cada atributo.

### 9.3 Salida

- Para cada rostro detectado:
  - Lista de atributos predichos con su score.
  - Formato estructurado (JSON o equivalente).
- Metadata: número de rostros detectados, tiempo de inferencia.

### 9.4 Formato de predicciones

```json
{
  "faces": [
    {
      "bbox": [x, y, w, h],
      "attributes": {
        "smiling": 0.92,
        "glasses": 0.15,
        "beard": 0.03
      }
    }
  ]
}
```

*Los nombres de atributos son ejemplos; los reales dependen del conjunto definido.*

### 9.5 Manejo de errores

- Reportar cuando no se detecte ningún rostro.
- Reportar cuando la calidad de la imagen sea insuficiente.
- No fallar silenciosamente.

### 9.6 Reproducibilidad

- Con los mismos inputs y configuración, la inferencia debe producir los mismos outputs.
- Versionar modelo y configuración utilizados.

---

## 10. Application / API

### Estado actual

**Pendiente de definición.** El repositorio aún no contempla una aplicación o API específica.

### Consideraciones futuras

- Si se implementa una API, debe seguir los principios de la constitución.
- Los endpoints y contratos se definirán cuando se tome la decisión de implementar una interfaz de servicio.

---

## 11. Configuration Module

### 11.1 Propósito

Centralizar la configuración del sistema para garantizar reproducibilidad y facilitar la gestión.

### 11.2 Configuraciones

```text
config/
├── pipeline.yaml      # Configuración general del pipeline
├── model.yaml         # Configuración del modelo
├── training.yaml      # Configuración de entrenamiento
├── inference.yaml     # Configuración de inferencia
└── datasets.yaml      # Configuración de datasets
```

### 11.3 Contenido por archivo

#### pipeline.yaml
- Rutas de directorios
- Parámetros generales
- Modo de operación (training/inference/evaluation)

#### model.yaml
- Arquitectura del modelo
- Hiperparámetros del modelo
- Configuración de entrada/salida

#### training.yaml
- Semilla aleatoria
- Hiperparámetros de entrenamiento (learning rate, batch size, épocas)
- Configuración de early stopping
- Métrica de selección de mejor modelo

#### inference.yaml
- Thresholds para predicciones binarias
- Configuración de face detection
- Opciones de optimización

#### datasets.yaml
- Lista de datasets disponibles
- Rutas y formatos
- Configuración de splits

### 11.4 Restricciones

- Las configuraciones deben ser versionadas.
- Los cambios de configuración deben documentarse.
- Las configuraciones deben ser validadas antes de usar.

---

## 12. Monitoring

### 12.1 Propósito

Seguimiento del rendimiento del modelo en producción para detectar degradación.

### 12.2 Métricas a monitorear

- **Distribución de predicciones**: cambios en la distribución de scores.
- **Volumen de predicciones**: cantidad de predicciones por período.
- **Latencia**: tiempo de respuesta del pipeline.
- **Errores**: tasa de errores y tipos de error.

### 12.3 Alertas

- Cambio significativo en distribución de predicciones.
- Incremento en tasa de errores.
- Degradación de latencia.

### 12.4 Trazabilidad

- Registro de predicciones realizadas.
- Asociación de predicciones con inputs.
- Historial de rendimiento.

### 12.5 Estado actual

**Pendiente de definición** si se requiere monitoreo activo desde el inicio.

---

## 13. Model Registry

### 13.1 Propósito

Almacenar, versionar y gestionar modelos entrenados.

### 13.2 Información por modelo

- Identificador único del modelo
- Versión
- Fecha de entrenamiento
- Métricas de evaluación
- Configuración utilizada
- Dataset utilizado
- Artefactos (modelo, tokenizer, etc.)

### 13.3 Estados del modelo

- **Development**: en desarrollo, no listo para uso.
- **Staging**: evaluado, pendiente de validación final.
- **Production**: validado y listo para uso.
- **Archived**: deprecado o reemplazado.

### 13.4 Operaciones

- Registrar nuevo modelo.
- Actualizar estado del modelo.
- Comparar modelos.
- Promocionar modelo a producción.

### 13.5 Estado actual

**Pendiente de definición** la implementación concreta (MLflow Model Registry o alternativa).

---

## 14. Retraining Pipeline

### 14.1 Propósito

Soporte para actualizaciones incrementales del modelo con nuevos datos.

### 14.2 Flujo

```text
new_data → validation → merge_with_existing → retraining → evaluation → model_registry
```

### 14.3 Pasos

1. **Validar nuevos datos**: verificar integridad y formato.
2. **Combinar con datos existentes**: merge controlado.
3. **Reentrenar modelo**: usar configuración existente.
4. **Evaluar**: comparar con modelo anterior.
5. **Registrar**: guardar nuevo modelo si cumple criterios.

### 14.4 Criterios de aceptación

- El nuevo modelo debe ser igual o mejor que el anterior.
- No debe haber regresión en métricas clave.
- La evaluación debe ser en conjunto de prueba independiente.

### 14.5 Estado actual

**Pendiente de definición** la frecuencia y criterios específicos de reentrenamiento.

---

## 15. Non-functional requirements

### 15.1 Reproducibilidad

- Cada pipeline (datos, training, evaluación, inferencia) debe poder ejecutarse de forma idéntica con la misma configuración.
- Versionar código, configuraciones y dependencias.

### 15.2 Testabilidad

- Cada componente debe poder testearse de forma aislada.
- Tests unitarios para lógica de negocio.
- Tests de integración para pipelines.

### 15.3 Mantenibilidad

- Código limpio, modular y bien documentado.
- Separación clara de responsabilidades.
- Type hints en todo el código Python.

### 15.4 Observabilidad

- Logging estructurado.
- Registro de métricas de rendimiento.
- Trazabilidad de predicciones.

### 15.5 Rendimiento

- La inferencia debe completarse en un tiempo razonable para su caso de uso.
- **Pendiente de definir** requisitos específicos de latencia o throughput.

### 15.6 Seguridad

- No exponer información sensible en el repositorio.
- Validar inputs para evitar inyección o ejecución arbitraria.

### 15.7 Privacidad

- Cumplir con los principios de la constitución sobre uso responsable.
- Minimizar datos personales almacenados.

---

## 16. Constraints

### Técnicas

- Python 3.11+ como lenguaje principal.
- PyTorch y torchvision para deep learning.
- scikit-learn para métricas y utilidades de ML.
- OpenCV y Pillow para procesamiento de imágenes.
- MLflow para tracking de experimentos.
- uv como gestor de paquetes.
- Docker para containerización.
- GitHub Actions para CI/CD.

### De diseño

- Separación estricta entre raw y processed data.
- Atributos exclusivamente visualmente observables.
- No inferir atributos sensibles.
- SDD como metodología de desarrollo.
- Separación de pipelines de training e inference.
- Configuración centralizada.

### Pendientes de definición

- Framework de explainability específico.
- Métricas y thresholds de evaluación.
- Arquitectura concreta del modelo.
- API o servicio de inferencia.
- Requisitos de rendimiento específicos.
- Implementación de Model Registry.
- Estrategia de monitoreo en producción.

---

*Última actualización: 2026-09-01*

---

## Resumen de cambios

### Nuevas secciones agregadas

| Sección | Nombre | Propósito |
|---------|--------|-----------|
| §11 | Configuration Module | Configuración centralizada del sistema |
| §12 | Monitoring | Seguimiento de rendimiento en producción |
| §13 | Model Registry | Almacenamiento y versionado de modelos |
| §14 | Retraining Pipeline | Soporte para actualizaciones incrementales |

### Secciones modificadas

| Sección | Cambios |
|---------|---------|
| §1 | Arquitectura general actualizada con diagrama completo |
| §2 | Soporte para múltiples datasets, validación como etapa obligatoria |
| §3 | Separación de pipelines training/inference |
| §5 | Soporte para landmarks y metadata adicional |
| §7 | Separación clara de evaluation, análisis de errores mejorado |
| §15 | Numeración actualizada |
| §16 | Constraints actualizados |
