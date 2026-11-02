# Instrucciones para agentes

Este archivo es el punto de entrada para agentes de IA que trabajen en este repositorio.

## Antes de actuar

1. Lee [`docs/architecture.md`](docs/architecture.md) antes de proponer o modificar código. Sus reglas de arquitectura y dominio son las referencias técnicas del proyecto.
2. Lee [`CONTRIBUTING.md`](CONTRIBUTING.md) antes de crear o reorganizar documentación, notebooks, ejemplos o trabajo compartido.
3. Revisa el estado de Git y el trabajo activo en `work-in-progress/` para evitar sobrescribir trabajo ajeno.
4. Si una instrucción del usuario contradice una regla documentada, señala el conflicto y sigue la instrucción explícita del usuario para esa tarea.

## Reglas de trabajo

- No cambies límites entre módulos ni convenciones físicas sin actualizar `docs/architecture.md` en el mismo cambio.
- No inventes una fuente de verdad paralela: código/API en `src/`, reglas técnicas en `docs/architecture.md`, flujo de colaboración en `CONTRIBUTING.md`.
- `examples/` y `examples/notebooks/` son responsabilidad exclusiva de Camilo. No modifiques, muevas ni reinicies su contenido salvo que Camilo lo pida explícitamente; puedes proponer cambios para que él los integre.
- Mantén cambios acotados al objetivo y no sobrescribas cambios preexistentes del usuario.
- No guardes secretos ni resultados binarios grandes en Git. Consulta `CONTRIBUTING.md` para compartir datos externos.
- Antes de entregar, informa qué cambió y qué validaciones ejecutaste. No afirmes que algo fue probado si no lo ejecutaste.

Estas instrucciones ayudan a orientar agentes que reconocen `AGENTS.md`; por sí solas no fuerzan a todos los proveedores o herramientas a leerlo. Las comprobaciones automáticas deben implementarse en CI cuando una regla pueda verificarse mecánicamente.
