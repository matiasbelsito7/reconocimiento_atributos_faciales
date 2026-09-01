# Constitution

Constitución técnica del proyecto **Facial Attribute Recognition**.

Define principios fundamentales e invariantes que guían todas las decisiones de desarrollo. Este documento es la autoridad de mayor nivel para decisiones arquitectónicas y técnicas.

---

## 1. Principios de arquitectura

- Separar claramente las responsabilidades: datos, preprocessing, face processing, training, evaluación, inferencia y aplicación.
- Mantener bajo acoplamiento y alta cohesión entre módulos.
- Preferir simplicidad y claridad antes que abstracciones prematuras.
- Diseñar para testabilidad: cada componente debe poder validarse de forma aislada.
- Evitar dependencias circulares.
- No imponer patrones arquitectónicos concretos salvo que exista una necesidad demostrada.
- La arquitectura debe poder evolucionar sin reescrituras mayores.
- Mantener configuración centralizada para garantizar reproducibilidad.
- Soportar múltiples datasets con diferentes características y formatos.

---

## 2. Principios de Machine Learning

- Reproducibilidad: cualquier experimento debe poder ejecutarse de forma idéntica con los mismos datos y configuración.
- Separar estrictamente datos de entrenamiento, validación y prueba.
- Versionar configuraciones y datasets cuando corresponda.
- Registrar experimentos con MLflow.
- Presentar métricas de forma honesta y completa; no seleccionar métricas que oculten debilidades del modelo.
- Analizar errores sistemáticamente antes de iterar sobre el modelo.
- Evitar data leakage: nunca utilizar información del conjunto de prueba en el entrenamiento.
- Mantener trazabilidad completa de transformaciones de datos.

---

## 3. Calidad de software

- Usar type hints en todo el código Python.
- Escribir tests para cambios relevantes.
- Cumplir las reglas de Ruff y el formato de Black.
- Ejecutar mypy para verificación de tipos estáticos.
- Mantener CI/CD funcional y no omitir checks.
- No ocultar errores para hacer pasar tests, linters o CI.
- No eliminar ni debilitar tests o herramientas de calidad sin justificación explícita.
- Revisar cambios antes de integrarlos.

---

## 4. Datos

- Separar datos raw de datos processed; nunca modificar archivos raw manualmente.
- Los pipelines de transformación deben ser reproducibles y versionados.
- Mantener la integridad de los datos originales.
- Validar datos antes de procesarlos.
- No incluir secretos, credenciales ni información sensible en el repositorio.
- Documentar la procedencia y licencia de los datasets utilizados.

---

## 5. Privacidad y uso responsable

- El sistema debe trabajar exclusivamente con **atributos faciales visualmente observables** (expresiones, accesorios, características faciales visibles).
- Está **explícitamente prohibido** utilizar el sistema para inferir atributos sensibles a partir del rostro, incluyendo pero no limitado a: raza, orientación sexual, discapacidad, estado de salud, afiliación política o religiosa.
- Aplicar minimización de datos: solo recopilar y almacenar la información estrictamente necesaria.
- Respetar las licencias de los datasets y bibliotecas utilizadas.
- Tratar imágenes y datos personales con responsabilidad y conforme a la normativa aplicable.

---

## 6. Documentación

Cada tipo de documento tiene una responsabilidad definida:

| Documento | Responsabilidad |
|-----------|-----------------|
| `AGENTS.md` | Comportamiento y contexto operativo de agentes |
| `docs/constitution.md` | Principios fundamentales (este documento) |
| `docs/specs.md` | Especificaciones funcionales y técnicas |
| `docs/tasks.md` | Tareas concretas de implementación |
| `docs/progreso/` | Registro de tareas completadas |
| `skills/*/SKILL.md` | Procedimientos y workflows especializados |
| `docs/changes/` | Registro histórico de cambios relevantes |

- No duplicar contenido entre documentos.
- Las decisiones importantes y cambios relevantes deben documentarse.
- Cada documento debe mantenerse dentro de su alcance definido.

---

## 7. Specification-Driven Development (SDD)

- Toda implementación debe estar respaldada por una especificación en `docs/specs.md`.
- Las tareas se derivan de las especificaciones, no de decisiones ad-hoc.
- Los cambios que alteren principios o arquitectura deben quedar documentados en `docs/changes/`.
- No introducir decisiones importantes únicamente de forma implícita en el código.
- Antes de implementar, consultar: constitución → specs → tareas → código existente.

---

*Última actualización: 2026-09-01*
