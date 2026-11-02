# UPVfab Design Tools

Herramientas Python para diseño y simulación de circuitos fotónicos integrados en UPVfab. El paquete separa las descripciones físicas (`core`), los backends de modos (`modesolver`), la propagación EME (`eme`) y las integraciones específicas de simulación (`tidy3d_plugin`).

## Empezar

El proyecto usa `uv` y Python 3.10 o posterior. Para instalar dependencias de desarrollo y del solver FEMWELL:

```bash
uv sync --extra modesolver --extra dev
```

Ejecuta un ejemplo desde la raíz del repositorio:

```bash
uv run python examples/core/test_waveguide.py
```

Las dependencias opcionales y flujos de instalación están definidos en [`pyproject.toml`](./pyproject.toml). Las simulaciones Tidy3D en la nube y sus estimaciones de coste deben activarse explícitamente; consulta la guía de arquitectura antes de añadir ese flujo a un ejemplo.

## Cómo está organizado

- [`src/upvfab_design_tools/`](./src/upvfab_design_tools/) contiene el paquete reutilizable.
- [`examples/`](./examples/) contiene material educativo para usuarios externos del paquete. Los notebooks están dentro de [`examples/notebooks/`](./examples/notebooks/). La estructura y el contenido actuales son transitorios: durante unas semanas se conservarán los ejemplos existentes y después la carpeta se reiniciará para dejar ejemplos seleccionados para usuarios externos.
- [`library/`](./library/) es el índice de resultados y referencias que el equipo considera útiles y suficientemente revisados para reutilizar.
- [`work-in-progress/`](./work-in-progress/) es el espacio temporal por persona para prototipos y notas aún no integrados.
- [`AGENTS.md`](./AGENTS.md) es el punto de entrada obligatorio para agentes de IA.
- [`docs/architecture.md`](./docs/architecture.md) registra reglas de arquitectura y convenciones físicas/de simulación.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) explica el flujo de colaboración y cómo promover resultados a la biblioteca.

## Fuentes de verdad

Para agentes, empieza por `AGENTS.md`; las reglas técnicas viven en `docs/architecture.md` y el flujo de equipo en `CONTRIBUTING.md`. Para el estado vigente de la API, la fuente es el código de `src/`. `examples/` es material de aprendizaje para usuarios externos y no es el espacio de trabajo colaborativo de la biblioteca. Los resultados de colaboración que deban conservarse se desarrollan en `work-in-progress/` y, una vez revisados, se documentan en `library/`.

## Datos de simulación

Los archivos grandes de resultados (por ejemplo HDF5) no se versionan automáticamente: `data/` está ignorado para evitar commits accidentales de artefactos pesados. Para compartirlos, acuerden un almacenamiento común y registren en la biblioteca el enlace/ubicación, nombre de archivo, fecha y parámetros de generación. Si un dataset pequeño debe versionarse, discútanlo antes y consideren Git LFS.

## Colaborar

Antes de cambiar interfaces o añadir un backend, revisa `AGENTS.md` y busca trabajo relacionado en `work-in-progress/`. Camilo mantiene `examples/` y `examples/notebooks/`; coordina con él cualquier cambio en esas carpetas. Usa una rama por cambio, mantén los cambios enfocados y enlaza resultados compartidos a su código de origen. Consulta [`CONTRIBUTING.md`](./CONTRIBUTING.md) para el flujo completo.
