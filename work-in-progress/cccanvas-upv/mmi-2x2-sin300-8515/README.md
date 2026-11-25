# MMI 2x2 SiN 300 nm, 85:15

- Responsable: cccanvas-upv
- Estado: en curso
- Actualizado: 2026-11-25
- Issue/PR: ninguno

## Objetivo

Reproducir el MMI SiN 2x2 85:15 con las funciones del paquete.

## Alcance y archivos

- 2x2_SiN300_8515_original.ipynb: copia del notebook original sin outputs.
- mmi2x2_sin300_8515_reproduction.ipynb: reproduccion y analisis.

## Reproduccion

Ejecutar el notebook desde la raiz; los datos HDF5 se guardan en data/ y no se versionan.

## Hallazgos y siguiente paso

El texto de origen menciona 114.71 um y un modelo dispersivo, pero el codigo ejecutado usa 113.5 um y SiN de indice constante 1.9729. La reproduccion toma el codigo como referencia.

## Resultado de reproduccion

El solver alcanzo el umbral de parada (1.24e-6). A 1.55 um: directa 14.6995 %, cruzada 85.3005 %, perdida 0.4411 dB y fase cruzada menos directa 89.85 grados. El criterio +/-0.01 se cumple en los 100 nm muestreados. Coste estimado 1.405653 FlexCredits. Datos: data/mmi2x2_sin300_8515_reproduction.hdf5, fuera de Git. La polarizacion de la fuente no se ha verificado independientemente.
