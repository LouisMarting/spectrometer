import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("MacOSX")
from matplotlib import colormaps

from spectrometer.filterbank.components import Filterbank, TransmissionLine, DirectionalFilter

plt.style.use('~/Repos/louis-style-docs/default.mplstyle')


### Basic filterbank settings
nF = int(5e4)
f = np.linspace(100e9,300e9,nF)

f0_min = 135e9
f0_max = 270e9
R_fb = 30
Ql = 22.6 # spacing at 30, but filter width (Ql) is 22.6
oversampling = R_fb/Ql

### Transmission lines
Z0_res = 44.9
eps_eff_res = 33.8
Qi_res = 1200
TL_res = TransmissionLine(Z0_res,eps_eff_res,Qi=Qi_res)

Z0_thru = 63.4
eps_eff_thru = 36.5
TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

Z0_kid = 63.4
eps_eff_kid = 36.5
TL_kid = TransmissionLine(Z0_kid,eps_eff_kid)


TransmissionLinesDict = {
    'through' : TL_thru,
    'resonator' : TL_res,
    'MKID' : TL_kid
}


## Filterbank
FB = Filterbank.from_parameters(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    R=R_fb,
    Ql=Ql,
    )

scale_down = 260/270
f0 = Filterbank.f0_range(f0_min=scale_down*f0_min,f0_max=scale_down*f0_max,R=R_fb)

FB = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0=f0,
    Ql=Ql
)

print(FB.R)
print(FB.oversampling)

# Caculate S-Parameters and realized values (suppress output)
FB.S(f);
FB.realized_parameters();



# plot filterbank
S31_all = FB.S31_absSq_list

cmap = colormaps['rainbow']
norm = mpl.colors.Normalize(vmin=0, vmax=len(FB.f0))

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(8,4),layout='constrained')

for i,S31_absSq in enumerate(S31_all.T):
    ax.plot(f/1e9,S31_absSq,color=cmap(norm(i)))

ax.plot(f/1e9,FB.S11_absSq,color='c',linestyle='--')
ax.plot(f/1e9,FB.S21_absSq,color='m',linestyle='--')
ax.plot(f/1e9,np.sum(FB.S31_absSq_list,axis=1),color='k',linestyle='--')

# plt.show()