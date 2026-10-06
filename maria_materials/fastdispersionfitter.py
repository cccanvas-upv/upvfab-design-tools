import matplotlib.pylab as plt
import numpy as np
import tidy3d as td

from tidy3d.plugins.dispersion import FastDispersionFitter, AdvancedFastFitterParam

## HACEMOS EL FITTING SOBRE LOS DATOS DE n, wvl, k DEL ELIPSO USANDO UNA HERRAMIENTA DE TIDY.

file_name_sio2 = "/home/maria_romero/upvfab-design-tools/maria_materials/raw_data/sio2_fastdispersionfitter_friendly.csv"
fitter_sio2 = FastDispersionFitter.from_file(file_name_sio2, skiprows= 1, delimiter = ";")

file_name_sin = "/home/maria_romero/upvfab-design-tools/maria_materials/raw_data/sin_fastdispersionfitter_friendly.csv"
fitter_sin = FastDispersionFitter.from_file(file_name_sin, skiprows= 1, delimiter = ";")

# fitter_sin.plot()
# plt.show()

# advanced_param = AdvancedFastFitterParam(weights=(1, 1))
# medium, rms_error = fitter_sio2.fit(max_num_poles=2, advanced_param=advanced_param, tolerance_rms=1e-4)
# fitter_sio2.plot(medium)
# plt.show()

# fname = "/home/maria_romero/upvfab-design-tools/maria_materials/fits/sio2.json"
# medium.to_file(fname)

# advanced_param = AdvancedFastFitterParam(weights=(1, 1))
# fitter_sio2_c = fitter_sio2.copy(update={"wvl_range": (1.5, 1.6)})
# sio2, rms_error = fitter_sio2_c.fit(max_num_poles=1, advanced_param=advanced_param, tolerance_rms=1e-4)
# fitter_sio2_c.plot(sio2)
# plt.show()

# fname = "/home/maria_romero/upvfab-design-tools/maria_materials/fits/sio2_cband.json"
# sio2.to_file(fname)

# advanced_param = AdvancedFastFitterParam(weights=(1, 1))
# medium, rms_error = fitter_sin.fit(max_num_poles=10, advanced_param=advanced_param, tolerance_rms=1e-4)
# fitter_sin.plot(medium)
# plt.show()

# fname = "/home/maria_romero/upvfab-design-tools/maria_materials/fits/sin_HEAVY.json"
# medium.to_file(fname)

advanced_param = AdvancedFastFitterParam(weights=(1, 1))
fitter_sin_c = fitter_sin.copy(update={"wvl_range": (1.5, 1.6)})
sin, rms_error = fitter_sin_c.fit(max_num_poles=2, advanced_param=advanced_param, tolerance_rms=1e-4)
fitter_sin_c.plot(sin)
plt.show()

# fname = "/home/maria_romero/upvfab-design-tools/maria_materials/fits/sin_cband.json"
# sin.to_file(fname)

