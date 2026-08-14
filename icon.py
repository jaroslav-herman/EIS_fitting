# %%

import wepy.basics as we
import wepy.eis as weis
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

ip = get_ipython()
if ip is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
# %%


name =  r"icon"
file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\453_VIII_cathode_etching_series_140min\VIII_Day2_Procedure1_PEIS_20_sccm_H2_C01.mpt"

data = we.read_file(file)
print(data)
groups = data.groupby('cycle number')

for cycle in [6,7,8,9]:
    f,Z,E,I = weis.freq_and_Z(data,cycle,(20000,20),control = 'Ewe-Ece')
    plt.plot(Z.real,-Z.imag,'--o',lw = 3,markersize = 10)

plt.text(0.5, 0.5, 'easy', transform=plt.gca().transAxes, fontsize=24, ha='center', va='center')
plt.gca().set_aspect("equal")
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)
plt.savefig(    name + ".png",    dpi=500,)
plt.savefig( name + ".svg",    dpi=500,)

plt.show()
# %%
