# MMI 2x2 SiN 300 nm con BCB, 85:15

- Responsable: cccanvas-upv
- Estado: en curso
- Actualizado: 2026-11-25
- Issue/PR: ninguno

## Objetivo

Reproducir el candidato de 113.71 um con tapers de 50 um del notebook original.

## Alcance y archivos

- sin300-bcb_original.ipynb: copia sin resultados guardados de la fuente.
- mmi2x2_sin300_bcb_8515_reproduction.ipynb: reproduccion con las funciones del paquete.

## Reproduccion

Ejecutar el notebook desde la raiz. Los datos HDF5 permanecen en data/, fuera de Git.

## Hallazgos y siguiente paso

La seleccion de 113.71 um y tapers de 50 um sigue la celda de parametros del notebook principal. El texto menciona 114.71 um. La fuente original usa indices constantes y el solver informo decaimiento final de 0.000116, superior a 1e-5.

## Resultado de reproduccion

El solver alcanzo el umbral de parada (3.89e-6). A 1.55 um: directa 14.7034 %, cruzada 85.2966 %, perdida 0.3241 dB y fase cruzada menos directa 89.58 grados. El criterio +/-0.01 se cumple en los 100 nm muestreados. Coste estimado 2.338585 FlexCredits. Datos: data/mmi2x2_sin300_bcb_8515_reproduction.hdf5, fuera de Git. La polarizacion de la fuente no se ha verificado independientemente. El plano modal del constructor se centra en z=0.20 um frente a z=0.15 um de la fuente original.
