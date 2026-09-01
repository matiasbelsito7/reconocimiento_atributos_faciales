Quiero que redactes `docs/specs.md`.

Antes de escribirlo, lee `AGENTS.md` y `docs/constitution.md`, y analiza el repositorio para entender el estado actual del proyecto.

## Objetivo

`specs.md` debe contener las **especificaciones funcionales y técnicas del sistema**.

Debe describir qué debe hacer el sistema, qué comportamiento se espera y qué restricciones técnicas concretas debe cumplir.

Debe ser lo suficientemente preciso como para que un agente pueda implementar una tarea sin tener que inferir requisitos importantes por su cuenta.

## Responsabilidad del documento

`specs.md` debe describir el **estado deseado del sistema**.

Debe responder preguntas como:

* ¿Qué funcionalidades debe proporcionar?
* ¿Qué entradas recibe?
* ¿Qué salidas produce?
* ¿Cómo debe comportarse cada componente?
* ¿Qué requisitos debe cumplir el pipeline?
* ¿Cómo se relacionan los diferentes componentes?
* ¿Qué condiciones y restricciones técnicas existen?

## Estructura sugerida

Organiza el documento de forma clara y modular. La estructura exacta puede adaptarse al proyecto, pero debería cubrir cuando corresponda:

### 1. System Overview

Descripción funcional del sistema y sus principales componentes.

### 2. Data

Especificar:

* entradas esperadas
* estructura y características relevantes de los datos
* requisitos de validación
* transformaciones esperadas
* separación entre datos originales y procesados
* requisitos de reproducibilidad y trazabilidad

### 3. Preprocessing

Especificar el comportamiento esperado del preprocessing de imágenes y cualquier etapa de preparación necesaria antes del modelo.

### 4. Face Processing

Especificar cómo debe tratarse la detección, extracción, normalización o preparación de rostros antes de la inferencia cuando corresponda.

### 5. Model

Definir los requisitos funcionales del modelo de reconocimiento de atributos:

* problema multilabel
* formato de entrada
* formato de salida
* representación de probabilidades/scores
* comportamiento esperado
* restricciones relevantes para entrenamiento e inferencia

No fijes una arquitectura concreta salvo que exista una decisión explícita que deba formar parte de la especificación.

### 6. Training

Especificar qué debe permitir el pipeline de entrenamiento:

* configuración reproducible
* datasets utilizados
* separación de datos
* tracking de experimentos
* checkpoints
* métricas
* manejo de modelos

### 7. Evaluation

Definir cómo debe evaluarse el sistema:

* métricas relevantes
* evaluación por atributo
* análisis de errores
* evaluación de calibración cuando corresponda
* evaluación de robustez cuando corresponda
* criterios para comparar modelos

No inventes thresholds o valores numéricos que todavía no hayan sido decididos.

### 8. Explainability

Especificar los requisitos para explicar las predicciones del modelo, cuando corresponda.

### 9. Inference

Definir el comportamiento esperado durante inferencia:

* entrada
* procesamiento
* salida
* formato de predicciones
* manejo de errores
* reproducibilidad cuando corresponda

### 10. Application / API

Si el repositorio ya contempla una aplicación o API, especificar su comportamiento esperado.

No inventes endpoints ni contratos que no estén respaldados por el código o por decisiones existentes.

### 11. Non-functional requirements

Especificar requisitos técnicos relevantes como:

* reproducibilidad
* testabilidad
* mantenibilidad
* observabilidad
* rendimiento
* seguridad
* privacidad

Solo incluir requisitos que realmente sean pertinentes al sistema.

### 12. Constraints

Registrar restricciones técnicas o de diseño que deban respetarse para implementar correctamente el sistema.

## Reglas importantes

No conviertas `specs.md` en una lista de tareas.

No incluir:

* tareas
* checklist de implementación
* roadmap
* commits
* comandos
* workflow de desarrollo
* instrucciones para agentes
* procedimientos de Git
* principios generales que ya pertenecen a `constitution.md`
* detalles innecesarios que cambien constantemente

Las tareas concretas pertenecen a `docs/tasks.md`.

Los principios fundamentales pertenecen a `docs/constitution.md`.

Las instrucciones operativas para agentes pertenecen a `AGENTS.md`.

Los procedimientos reutilizables pertenecen a `skills/*/SKILL.md`.

El historial de cambios pertenece a `docs/changes/`.

## Precisión

No inventes requisitos.

Cuando una decisión todavía no esté definida, no la presentes como requisito definitivo. En esos casos:

* omítela, si no es necesaria para la especificación actual, o
* deja explícitamente indicada la decisión como pendiente de definición.

Distingue entre:

* requisitos ya establecidos
* comportamiento observable esperado
* decisiones técnicas que todavía no están fijadas

## SDD

Las especificaciones deben servir como fuente de verdad para las tareas de implementación.

Las tareas futuras de `docs/tasks.md` deben poder derivarse de las especificaciones sin necesidad de reinterpretar el objetivo del sistema.

## Restricciones

No modifiques ningún otro archivo.

Solo crea o modifica:

`docs/specs.md`

Antes de finalizar, verifica que:

1. no duplique innecesariamente `AGENTS.md`
2. no contradiga `docs/constitution.md`
3. no contenga tareas
4. no invente decisiones técnicas todavía no tomadas
5. describa claramente el comportamiento esperado del sistema

Al terminar, dame un resumen breve de las secciones creadas y de cualquier requisito importante que haya quedado explícitamente pendiente de definición.
