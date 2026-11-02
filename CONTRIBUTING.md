# Contribuir

Esta guía describe cómo trabajar en equipo y mantener resultados útiles accesibles. La arquitectura y las convenciones técnicas están en [`docs/architecture.md`](docs/architecture.md); las instrucciones para agentes están en [`AGENTS.md`](AGENTS.md).

## Flujo de trabajo

1. Sincroniza la rama base y revisa las instrucciones anteriores, el estado de Git y `work-in-progress/`.
2. Crea una rama por tarea. Anota en el README de tu carpeta de trabajo el alcance, archivos principales y dependencias con otros cambios.
3. Mantén prototipos temporales en `work-in-progress/<identificador>/`. Usa un identificador estable acordado por el equipo (por ejemplo, usuario de GitHub). Copia [`work-in-progress/_template/README.md`](work-in-progress/_template/README.md).
4. El trabajo colaborativo de investigación y desarrollo permanece en `work-in-progress/` hasta que se revise. `examples/` y `examples/notebooks/` son material educativo dirigido a usuarios externos.
5. Si un resultado de colaboración es reutilizable, promuévelo a `library/` con una ficha revisada que explique el resultado y permita reproducirlo. Abre un PR enfocado que explique decisión, evidencia, reproducción y cualquier cambio de API; luego retira o archiva el prototipo.
6. No edites silenciosamente el prototipo de otra persona; coordina una transferencia o colaboración explícita.

## Dónde guardar cada cosa

- `src/`: API e implementación reutilizable.
- `examples/`: ejemplos educativos dirigidos a usuarios externos del paquete. Camilo es el único responsable de mantener e integrar cambios en esta carpeta; otras personas pueden proponerlos para que él los integre.
- `examples/notebooks/`: notebooks educativos y demostraciones bajo la misma responsabilidad exclusiva de Camilo. No es el espacio de trabajo de quienes colaboran en resultados de la biblioteca.
- `library/`: resultados revisados de la colaboración, presentados para que también puedan consultarlos usuarios externos. Es un índice/ficha de resultados, no un duplicado de ejemplos.
- `work-in-progress/`: prototipos temporales organizados por persona.

Una ficha de biblioteca debe indicar propósito, conclusión, supuestos, límites, pasos de reproducción y ubicación de datos externos cuando aplique. Enlaza a código fuente o datos de soporte sin duplicar materiales educativos de `examples/`.

Los notebooks de `examples/` deben minimizar outputs pesados, eliminar rutas locales y estado oculto, y registrar unidades, materiales, dimensiones, longitud de onda, backend, resolución y versión/commit relevantes para reproducir cifras.

## Transición de examples

Durante unas semanas se conservará el contenido existente de `examples/` mientras se reorganiza el repositorio. Ese contenido es legado/transitorio y puede incluir scripts de validación o trabajo interno que no representa todavía la experiencia educativa final. Después de ese periodo, Camilo reiniciará la carpeta y dejará allí únicamente ejemplos seleccionados y mantenidos para usuarios externos del paquete. No uses `examples/` como espacio de trabajo colaborativo ni asumas que el contenido actual permanecerá.

## Datos y seguridad

`data/` y los archivos HDF5 están ignorados por Git para evitar subir artefactos pesados accidentalmente. Comparte esos datos mediante almacenamiento común y documenta ubicación accesible, nombre, fecha y parámetros de generación. Considera Git LFS si el equipo acuerda versionar un dataset.

No guardes API keys, tokens, `.env`, credenciales ni capturas con secretos en Git. Las claves para servicios externos se configuran según `docs/architecture.md`; las ramas locales deben mantener sus credenciales fuera del repositorio.

## Revisión y automatización

Antes de abrir un PR, revisa el diff y elimina outputs, archivos temporales y datos no intencionados. En el PR explica qué cambió y qué validación ejecutaste. Las reglas que puedan comprobarse mecánicamente deben añadirse a CI; la documentación y las listas de revisión orientan, pero no garantizan por sí solas el cumplimiento.
