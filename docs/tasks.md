# Tasks

Tareas concretas de implementación derivadas de `docs/specs.md`.

Cada tarea referencia la sección de especificación que la respalda. Las decisiones pendientes se marcan explícitamente.

---

## Fase 0: Infraestructura del proyecto

### T-0.1: Crear estructura de directorios
- **Especificación**: specs §16 (Constraints)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Crear estructura base del proyecto: `src/`, `data/raw/`, `data/processed/`, `docs/`, `tests/`, `notebooks/`, `config/`.

### T-0.2: Configurar `pyproject.toml`
- **Especificación**: specs §16 (Constraints)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Definir proyecto Python con dependencias principales (torch, torchvision, scikit-learn, opencv-python, pillow, mlflow, pytest, ruff, black, mypy).

### T-0.3: Crear `Makefile`
- **Especificación**: specs §16 (Constraints)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Documentar comandos principales: sync, test, lint, format, typecheck, pre-commit.

### T-0.4: Configurar `.gitignore`
- **Especificación**: constitution §4 (Datos), constitution §5 (Privacidad)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Excluir datos raw, modelos entrenados, artefactos MLflow, secretos, archivos temporales.

### T-0.5: Configurar `pre-commit`
- **Especificación**: constitution §3 (Calidad de software)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Configurar hooks de Ruff, Black, mypy y validaciones básicas.

### T-0.6: Configurar CI/CD básico
- **Especificación**: constitution §3 (Calidad de software), specs §15.1 (Reproducibilidad)
- **Prioridad**: media
- **Estado**: completada
- **Descripción**: GitHub Actions para lint, format check, typecheck y tests.

---

## Fase 1: Data

### T-1.1: Seleccionar dataset
- **Especificación**: specs §2.2 (Entradas esperadas), specs §2.4 (Atributos soportados)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Seleccionar dataset de atributos faciales visualmente observables. Documentar fuente, licencia y tamaño. Definir lista final de atributos.

### T-1.2: Definir estructura de anotaciones
- **Especificación**: specs §2.3 (Estructura de datos)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Crear CSV con identificador de imagen y columnas binarias por atributo.

### T-1.3: Script de validación de datos
- **Especificación**: specs §2.6 (Validación de datos)
- **Prioridad**: alta
- **Estado**: completada
- **Descripción**: Verificar integridad de imágenes, formato de anotaciones, duplicados y distribución de atributos.

### T-1.4: Gestor de datasets
- **Especificación**: specs §2.5 (Separación de datos)
- **Prioridad**: media
- **Estado**: completada
- **Descripción**: Implementar DatasetManager con soporte para múltiples datasets y división train/val/test.

### T-1.5: Documentar dataset
- **Especificación**: specs §2.7 (Reproducibilidad y trazabilidad)
- **Prioridad**: media
- **Estado**: completada
- **Descripción**: Crear registro con fuente, licencia, fecha de obtención, tamaño y atributos incluidos.

### T-1.6: Soporte para múltiples datasets
- **Especificación**: specs §2.1 (Gestión de múltiples datasets)
- **Prioridad**: media
- **Estado**: completada
- **Descripción**: Implementar estructura para soportar múltiples datasets con diferentes características.

### T-1.7: Versionado de datos
- **Especificación**: specs §2.8 (Versionado de datos)
- **Prioridad**: baja
- **Estado**: pendiente
- **Descripción**: Implementar sistema de versionado para datos procesados.

---

## Fase 2: Preprocessing

### T-2.1: Pipeline de preprocessing para entrenamiento
- **Especificación**: specs §3.1 (Preprocessing para entrenamiento)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Implementar redimensionamiento, normalización de color, corrección básica y data augmentation. Mantener trazabilidad de transformaciones.

### T-2.2: Pipeline de preprocessing para inferencia
- **Especificación**: specs §3.2 (Preprocessing para inferencia)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Implementar pipeline optimizado para latencia baja, sin augmentation, para imagen individual.

### T-2.3: Tests de preprocessing
- **Especificación**: constitution §3 (Calidad), specs §3.4 (Restricciones)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Tests para verificar reproducibilidad y que no se degrada calidad innecesariamente en ambos pipelines.

---

## Fase 3: Face Processing

### T-3.1: Seleccionar detector de rostros
- **Especificación**: specs §4.1 (Detección de rostros)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Evaluar opciones (MTCNN, RetinaFace, Haar Cascade, HOG). **Decisión pendiente.**

### T-3.2: Implementar detección de rostros
- **Especificación**: specs §4.1 (Detección de rostros)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Detectar bounding box de rostros, manejar múltiples rostros por imagen.

### T-3.3: Implementar extracción de rostros
- **Especificación**: specs §4.2 (Extracción)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Recortar región del rostro, mantener relación de aspecto.

### T-3.4: Implementar normalización
- **Especificación**: specs §4.3 (Normalización)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Redimensionar a tamaño uniforme, normalizar píxeles.

### T-3.5: Manejo de errores en face processing
- **Especificación**: specs §4.4 (Manejo de errores)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Reportar imágenes sin rostro detectado, decidir comportamiento para baja calidad.

### T-3.6: Tests de face processing
- **Especificación**: constitution §3 (Calidad)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Tests para detección, extracción, normalización y manejo de errores.

---

## Fase 4: Model

### T-4.1: Seleccionar arquitectura del modelo
- **Especificación**: specs §5.5 (Decisiones pendientes)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Evaluar opciones (ResNet, EfficientNet, custom CNN). **Decisión pendiente.**

### T-4.2: Implementar modelo
- **Especificación**: specs §5.1-5.4 (Model)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Implementar modelo multilabel con entrada rostro normalizado y salida vector de scores [0,1].

### T-4.3: Seleccionar función de pérdida
- **Especificación**: specs §5.5 (Decisiones pendientes)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Evaluar BCE, BCE con pesos, etc. **Decisión pendiente.**

### T-4.4: Tests del modelo
- **Especificación**: constitution §3 (Calidad)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Tests para verificar forma de entrada/salida, rango de scores, comportamiento independiente por atributo.

---

## Fase 5: Training

### T-5.1: Configuración reproducible
- **Especificación**: specs §6.1 (Configuración)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Sistema de configuración con semillas, hiperparámetros y versiones de dependencias.

### T-5.2: Pipeline de entrenamiento
- **Especificación**: specs §6.2-6.3 (Datasets, Separación)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Entrenamiento con splits estrictos (train/val/test), evitar data leakage.

### T-5.3: Tracking con MLflow
- **Especificación**: specs §6.4 (Tracking)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Registrar configuración, métricas, artefactos y duración de cada experimento.

### T-5.4: Sistema de checkpoints
- **Especificación**: specs §6.5 (Checkpoints)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Guardar checkpoints periódicos, reanudar entrenamiento, mantener mejor modelo.

### T-5.5: Métricas de entrenamiento
- **Especificación**: specs §6.6 (Métricas de entrenamiento)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Registrar métricas por época, detectar overfitting/underfitting.

### T-5.6: Tests de training
- **Especificación**: constitution §3 (Calidad)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Tests para verificar reproducibilidad, splits correctos y tracking.

---

## Fase 6: Evaluation

### T-6.1: Definir métricas de evaluación
- **Especificación**: specs §7.1 (Métricas relevantes)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Seleccionar métricas multilabel (exactitud, precisión, recall, F1 por atributo, métricas globales). **Decisión pendiente.**

### T-6.2: Implementar evaluación por atributo
- **Especificación**: specs §7.2 (Evaluación por atributo)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Reportar rendimiento de cada atributo individualmente.

### T-6.3: Análisis de errores
- **Especificación**: specs §7.3 (Análisis de errores)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Identificar patrones, atributos confundidos, visualizar errores.

### T-6.4: Comparación de modelos
- **Especificación**: specs §7.6 (Criterios de comparación)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Framework para comparar modelos usando mismas métricas y conjunto de prueba.

### T-6.5: Tests de evaluación
- **Especificación**: constitution §3 (Calidad)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Tests para verificar cálculo correcto de métricas.

---

## Fase 7: Inference

### T-7.1: Pipeline de inferencia
- **Especificación**: specs §9.1-9.3 (Inference)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Preprocessing → Face processing → Predicción → Salida estructurada.

### T-7.2: Formato de salida
- **Especificación**: specs §9.4 (Formato de predicciones)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Implementar salida JSON con bounding boxes y scores por atributo.

### T-7.3: Manejo de errores en inferencia
- **Especificación**: specs §9.5 (Manejo de errores)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Reportar errores sin fallar silenciosamente.

### T-7.4: Tests de inferencia
- **Especificación**: constitution §3 (Calidad)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Tests de integración del pipeline completo.

---

## Fase 8: Configuration Module

### T-8.1: Diseñar estructura de configuración
- **Especificación**: specs §11 (Configuration Module)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Definir estructura de archivos de configuración (pipeline.yaml, model.yaml, training.yaml, inference.yaml, datasets.yaml).

### T-8.2: Implementar sistema de configuración
- **Especificación**: specs §11 (Configuration Module)
- **Prioridad**: alta
- **Estado**: pendiente
- **Descripción**: Implementar carga y validación de configuraciones desde archivos YAML.

### T-8.3: Tests de configuración
- **Especificación**: constitution §3 (Calidad)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Tests para verificar carga, validación y reproducibilidad de configuraciones.

---

## Fase 9: Monitoring

### T-9.1: Diseñar sistema de monitoreo
- **Especificación**: specs §12 (Monitoring)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Definir métricas a monitorear, alertas y formato de registro.

### T-9.2: Implementar registro de predicciones
- **Especificación**: specs §12.4 (Trazabilidad)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Implementar logging de predicciones realizadas con metadata asociada.

### T-9.3: Implementar alertas básicas
- **Especificación**: specs §12.3 (Alertas)
- **Prioridad**: baja
- **Estado**: pendiente
- **Descripción**: Implementar alertas para cambios significativos en distribución de predicciones.

---

## Fase 10: Model Registry

### T-10.1: Diseñar Model Registry
- **Especificación**: specs §13 (Model Registry)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Definir estados del modelo, información a almacenar y operaciones soportadas.

### T-10.2: Implementar Model Registry básico
- **Especificación**: specs §13 (Model Registry)
- **Prioridad**: media
- **Estado**: pendiente
- **Descripción**: Implementar registro, versionado y consulta de modelos entrenados.

### T-10.3: Integrar con MLflow
- **Especificación**: specs §13 (Model Registry)
- **Prioridad**: baja
- **Estado**: pendiente
- **Descripción**: Evaluar uso de MLflow Model Registry como backend.

---

## Fase 11: Retraining Pipeline

### T-11.1: Diseñar pipeline de reentrenamiento
- **Especificación**: specs §14 (Retraining Pipeline)
- **Prioridad**: baja
- **Estado**: pendiente
- **Descripción**: Definir flujo de validación → merge → reentrenamiento → evaluación.

### T-11.2: Implementar merge de datasets
- **Especificación**: specs §14.3 (Pasos)
- **Prioridad**: baja
- **Estado**: pendiente
- **Descripción**: Implementar combinación controlada de datasets existentes con nuevos datos.

### T-11.3: Implementar criterios de aceptación
- **Especificación**: specs §14.4 (Criterios de aceptación)
- **Prioridad**: baja
- **Estado**: pendiente
- **Descripción**: Implementar comparación automática con modelo anterior.

---

## Decisiones pendientes resumen

| Decisión | Fase afectada | Estado |
|----------|---------------|--------|
| Dataset a utilizar | Fase 1 | Pendiente |
| Lista final de atributos | Fase 1 | Pendiente |
| Detector de rostros | Fase 3 | Pendiente |
| Arquitectura del modelo | Fase 4 | Pendiente |
| Función de pérdida | Fase 4 | Pendiente |
| Métricas de evaluación | Fase 6 | Pendiente |
| Interfaz de aplicación | Fase 8 | Pendiente |
| Estrategia de monitoreo | Fase 9 | Pendiente |
| Implementación de Model Registry | Fase 10 | Pendiente |
| Frecuencia de reentrenamiento | Fase 11 | Pendiente |

---

*Última actualización: 2026-09-01*
