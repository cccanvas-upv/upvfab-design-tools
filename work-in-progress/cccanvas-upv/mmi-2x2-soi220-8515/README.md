# MMI 2x2 SOI 220 nm, 85:15

- Responsable: cccanvas-upv
- Estado: en curso
- Actualizado: 2026-11-25
- Issue/PR: ninguno

## Objetivo

Reproducir el diseno SOI 2x2 85:15 usando las funciones del paquete.

## Alcance y archivos

- 2x2_SOI220_8515_original.ipynb: copia sin outputs del notebook de origen.
- mmi2x2_soi220_8515_reproduction.ipynb: modelo y analisis con datos Tidy3D.

## Reproduccion

Ejecutar el notebook de reproduccion desde la raiz. Los datos se guardan en data/ y no se incluyen en Git.

## Hallazgos y siguiente paso

Se conserva la geometria de la simulacion base (39.60 um, tapers de 5 um). El notebook del barrido usa otras longitudes de taper y contiene una descripcion textual de longitud base inconsistente con sus parametros.

## Resultado de reproduccion

La simulacion alcanzo el umbral de parada (9.11e-6). A 1.55 um: directa 14.8164 %, cruzada 85.1836 %, perdida 0.3637 dB y fase cruzada menos directa 89.72 grados. Ancho de banda muestreado alrededor de 1.55 um: 64 nm con tolerancia +/-0.01. Coste estimado 1.373311 FlexCredits. Datos: data/mmi2x2_soi220_8515_reproduction.hdf5, fuera de Git.
