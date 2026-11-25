# MMI 2x2 SOI 220 nm, 50:50

- Responsable: cccanvas-upv
- Estado: exploracion
- Actualizado: 2026-11-25
- Issue/PR: ninguno

## Objetivo

Reproducir el MMI 2x2 SOI 220 nm 50:50 mediante las funciones del paquete y preparar su ficha de biblioteca.

## Alcance y archivos

- 2x2_SOI220_5050_original.ipynb: copia del notebook de origen sin resultados guardados.
- mmi2x2_soi220_5050_reproduction.ipynb: modelo, carga y graficas de la reproduccion.
- El HDF5 data/soi_5050_fdtd.hdf5 corresponde a otra geometria y no se usa como resultado de este dispositivo.

## Reproduccion

La fuente original usa longitud 26.8 um, ancho 4.8 um y tapers de 1.3 um; excita la entrada superior. Los datos FDTD se guardan fuera de Git, en data/.

## Hallazgos y siguiente paso

El resultado original advierte decaimiento final 0.00288 frente al umbral 1e-5. La reproduccion con el constructor del paquete tiene diferencias en fuente y monitores que deben documentarse y compararse.

## Resultado de reproduccion

La simulacion alcanzo el umbral de parada (9.92e-6). A 1.55 um: directa 51.5165 %, cruzada 48.4835 %, perdida 0.1345 dB y fase cruzada menos directa 89.38 grados. El coste estimado fue 1.144729 FlexCredits. Datos locales: data/mmi2x2_soi220_5050_reproduction.hdf5 (ignorado por Git). Notebook de biblioteca: ../../../library/mmis/mmi2x2_soi220_5050.ipynb.
