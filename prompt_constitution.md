Quiero que redactes `docs/constitution.md` para este proyecto siguiendo un enfoque de **Specification-Driven Development (SDD)**.

Antes de escribirlo, analiza el `AGENTS.md` existente y la estructura actual del repositorio para entender el contexto del proyecto.

## Objetivo

`constitution.md` debe definir los **principios fundamentales e invariantes del proyecto** que deben guiar todas las decisiones futuras de desarrollo.

Debe funcionar como una constitución técnica: establecer reglas de alto nivel que permanezcan relativamente estables durante la evolución del proyecto.

## Información del proyecto

El proyecto es **Facial Attribute Recognition**, un proyecto de Computer Vision / Data Science orientado al reconocimiento multilabel de atributos faciales visualmente observables mediante Machine Learning y Deep Learning.

Stack principal actual:

* Python 3.11+
* PyTorch
* torchvision
* scikit-learn
* NumPy
* pandas
* OpenCV
* Pillow
* MLflow
* pytest
* Ruff
* Black
* mypy
* pre-commit
* uv
* Docker
* GitHub Actions

La arquitectura general separa:

```text
data
→ preprocessing
→ face processing
→ training
→ models
→ evaluation
→ inference
→ application
```

El proyecto utiliza SDD y mantiene separadas estas responsabilidades documentales:

```text
AGENTS.md
→ comportamiento y contexto operativo de los agentes

docs/constitution.md
→ principios fundamentales del proyecto

docs/specs.md
→ especificaciones funcionales y técnicas

docs/tasks.md
→ tareas concretas de implementación

skills/*/SKILL.md
→ procedimientos y workflows especializados

docs/changes/
→ registro histórico de cambios relevantes
```

## Qué debe contener

La constitución debería cubrir, como mínimo:

### 1. Principios de arquitectura

Definir principios como:

* separación clara de responsabilidades
* modularidad
* bajo acoplamiento y alta cohesión
* testabilidad
* mantenibilidad
* simplicidad frente a sobreingeniería
* evolución controlada de la arquitectura

No imponer patrones concretos si no son necesarios.

### 2. Principios de Machine Learning

Establecer principios sobre:

* reproducibilidad
* separación entre datos, entrenamiento, evaluación e inferencia
* evaluación rigurosa
* trazabilidad de experimentos
* versionado de configuraciones/datasets cuando corresponda
* honestidad en la presentación de métricas
* análisis de errores
* evitar data leakage

### 3. Calidad de software

Definir expectativas sobre:

* type hints
* tests
* linting
* formatting
* static type checking
* CI/CD
* revisión de cambios
* código mantenible

### 4. Datos

Establecer principios sobre:

* separación entre raw y processed data
* reproducibilidad de pipelines
* integridad de los datos originales
* validación de datos
* no incluir secretos en el repositorio
* trazabilidad de transformaciones

### 5. Privacidad y uso responsable

El sistema debe trabajar exclusivamente con **atributos faciales visualmente observables**.

La constitución debe prohibir explícitamente utilizar el sistema para inferir atributos sensibles a partir del rostro.

También debe establecer principios de minimización de datos, respeto de licencias y tratamiento responsable de imágenes y datasets.

### 6. Documentation

Establecer que las decisiones importantes y cambios relevantes deben ser documentados.

Distinguir claramente:

* principios → constitution
* especificaciones → specs
* tareas → tasks
* historial de cambios → `docs/changes/`

### 7. SDD

Definir que:

* las implementaciones deben estar respaldadas por especificaciones
* las tareas deben derivarse de las especificaciones
* los cambios que alteren principios o arquitectura deben quedar documentados
* no se deben introducir decisiones importantes únicamente de forma implícita en el código

## Qué NO debe contener

Es muy importante que `constitution.md` **no se convierta en un documento que contenga toda la documentación del proyecto**.

No debe incluir:

* lista detallada de tareas
* roadmap
* comandos
* procedimientos paso a paso
* instrucciones detalladas para los agentes
* implementación concreta de módulos
* endpoints específicos
* hiperparámetros concretos
* detalles que cambien frecuentemente
* instrucciones de commit/push
* workflow detallado de desarrollo

Ese contenido pertenece a `tasks.md`, `specs.md`, `AGENTS.md` o las `SKILL.md` correspondientes.

## Estilo

El documento debe ser:

* claro
* conciso
* normativo
* técnico
* fácil de consultar por agentes
* suficientemente general para mantenerse estable durante el proyecto

Evita lenguaje excesivamente burocrático.

Cada principio debería poder interpretarse como una regla que permita decidir entre dos alternativas de implementación.

Por ejemplo:

> "Preferir simplicidad y claridad antes que abstracciones prematuras."

es mejor que una explicación extensa sobre diseño de software.

## Importante

No modifiques otros archivos.

Solo crea o modifica:

```text
docs/constitution.md
```

Al terminar, revisa el documento para asegurarte de que no duplica innecesariamente responsabilidades de `AGENTS.md`, `specs.md`, `tasks.md` o `skills/*/SKILL.md`.

Después muéstrame un resumen breve de los principios que definiste y de las posibles decisiones que dejaste deliberadamente fuera de la constitución.
