# Trabajo: Reproducción MMI 1×2 SOI 220 nm, 50:50

- **Responsable:** Camilo Cano
- **Estado:** En curso
- **Actualizado:** 2026-11-25
- **Issue/PR relacionado:** ninguno

## Objetivo

Reproducir el diseño y los resultados del notebook original con la API de upvfab_design_tools y preparar una ficha de biblioteca tras validar la reproducción.

## Alcance y archivos

- 1x2_SOI220_5050_original.ipynb: copia local sin outputs históricos ni contadores de ejecución.
- mmi1x2_soi220_5050_reproduction.ipynb: prototipo de geometría, conversión de materiales, simulación opcional y gráficas.
- No se modifica el repositorio remoto ni se promueven resultados antes de validarlos.
- La API actual no incluye constructor MMI 1×2; se reutilizan conversión de estructuras y plotting FDTD.

## Reproducción

Abrir el notebook de reproducción desde la raíz del repositorio con las dependencias tidy3d y matplotlib. Las celdas cloud están desactivadas; activarlas requiere credenciales y puede generar coste.

La referencia usa n(Si)=3.48 y n(SiO₂)=√2.085136 constantes. El prototipo usa SILICON y THERMAL_SILICON_DIOXIDE del paquete; considerar esa diferencia al comparar cifras.

## Hallazgos y siguiente paso

El notebook ya incluye salidas de figura para la planta del MMI y la seccion de materiales, generadas localmente con utilidades de visualizacion del paquete. La seccion 7 ahora usa dos paneles lado a lado como la figura de la biblioteca: ratio 50:50 con umbral de ?1 % y ancho de banda continuo alrededor de 1.55 ?m, y perdida en exceso. Las graficas de campo y barrido requieren sim_data de Tidy3D; falta validar la geometria y comparar una simulacion con las cifras de referencia. No declarar resultados numericos como reproducidos antes de esa comparacion.
