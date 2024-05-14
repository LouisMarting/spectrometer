#%%
## Make a kidlist based on input from the filterbank_geometry.csv
## Author: Louis Marting
## Date: December 2023

# Importing packages
import matplotlib
matplotlib.use('QtAgg')
import numpy as np

from filterbank.components import TransmissionLine, Coupler
from filterbank.transformations import chain, Zin_from_abcd, abcd_shuntload,abcd2s
from matplotlib import pyplot as plt
from kidencoding import shuffle_kids, shuffle_kids2
import pandas as pd

# Physical constants
mu0 = np.pi * 4e-7
eps0 = 8.854187817620e-12
c0 = 1 / np.sqrt(eps0 * mu0)


# Read filterbank geometry data
filterbank_geometry = np.genfromtxt("filterbank_geometry.csv",delimiter=",",skip_header=1)
n_filters = np.size(filterbank_geometry, axis=0)



### USER SETTINGS ###

# KID settings
kid_Qc = 40e3 # 40k
fr_max_at_lhybrid_min = 6e9
l_hybrid_min = 1.2e-3 # in [meter], @ 6GHz. minimum hybrid section length at the shorted end (connected to port 2)
# in code below: try to keep alu fraction constant
fraction_l_hybrid_shorted = 0.67


# Readout settings
fr_min = 4.135e9
f_lo = 5e9
f_lo_gap = 0.12e9
fr_max = 5.865e9
group_size = 7
width_MS_MKID = 1.45e-6 #1.45 um wide microstrip that couples to filter (part of MKID)
width_MS_RES = 1.0e-6 #1.0 um wide microstrip of the filter

# Set df = 4.1 MHz at 5 GHz. 
# geometric compression/expansion of that gap.
relative_spacing = 4.1e6 / 5e9



# Define KID TL parameters
TL_NbTiN = TransmissionLine(Z0=73.7,eps_eff=9.5)
TL_Hybrid = TransmissionLine(Z0=71.0,eps_eff=8.94,Qi=1e6)
TL_Microstrip = TransmissionLine(Z0=65.3,eps_eff=28.9)

Z0_thru_RO_line = 63.8

### END USER SETTINGS ###

# %%



def geometric_frequency_spacing(f0_min,f0_max,relative_spacing):
    """
        space frequencies geometrically so that they have a constant relative spacing.

        Input:
            f0_min
            f0_max
            relative_spacing (delta_f)/f
    """
    n_filters = int(np.floor(1 + np.log10(f0_max / f0_min) / np.log10(1 + 1 * relative_spacing)))
    
    f0 = np.zeros(n_filters)
    f0[0] = f0_min
    for i in np.arange(1,n_filters):
        f0[i] = f0[i-1] + f0[i-1] * relative_spacing
    return f0



fr_lower = geometric_frequency_spacing(fr_min,f_lo-f_lo_gap/2,relative_spacing)
fr_upper = geometric_frequency_spacing(f_lo+f_lo_gap/2,fr_max,relative_spacing)


# Get n_upper, n_lower given the geometric expansion.
# Ensure that the n_upper + n_lower >= n_filters
print(f"n in lower band: {len(fr_lower)}")
print(f"n in upper band: {len(fr_upper)}")
if (len(fr_lower) + len(fr_upper)) < n_filters:
    raise ValueError("cannot fit enough filters")

# Remove the extra filters but make sure it are only a few
n_extra = (len(fr_lower) + len(fr_upper)) - n_filters
if n_extra > 10:
    raise Warning("The relative spacing might be too close, consider reducing requirement")

n_remove_lower = int(np.ceil(n_extra/2))
n_remove_upper = int(n_extra - n_remove_lower)

fr_lower = fr_lower[n_remove_lower:]
fr_upper = fr_upper[:-n_remove_upper]



# shuffle
n_lower_sideband, n_upper_sideband = (len(fr_lower),len(fr_upper))

print(f"N filters in lower sideband : {n_lower_sideband}")
print(f"N filters in upper sideband : {n_upper_sideband}")

# 215 filters in lower and 178 in upper for D2.1

# shuffle_l_i = shuffle_kids2(n_lower_sideband,[[6,8],[7,9],[7,8],[6,8]])
shuffle_l_i = shuffle_kids(n_lower_sideband,group_size)
fr_lower = fr_lower[shuffle_l_i]
shuffle_l = shuffle_l_i + 100

# shuffle_u_i = shuffle_kids2(n_upper_sideband,[[8,5],[7,7],[7,7],[8,5]])
shuffle_u_i = shuffle_kids(n_upper_sideband,group_size)
fr_upper = fr_upper[shuffle_u_i]
shuffle_u = shuffle_u_i + n_lower_sideband + 100

shuffle = np.concatenate((shuffle_l, shuffle_u))
fr = np.concatenate((fr_lower,fr_upper)) # This



# def split_filterbank_evenly(n_filters):
#     # Divide filterbank in two, with lower sideband 1 more if uneven.
#     n_upper = n_filters//2
#     n_lower = n_filters - n_upper
#     return n_lower, n_upper

# n_lower_sideband, n_upper_sideband = split_filterbank_evenly(n_filters)

# shuffle_l_i = shuffle_kids(n_lower_sideband,7)
# fr_lower = np.linspace(fr_min,f_lo-f_lo_gap/2,n_lower_sideband)[shuffle_l_i]
# shuffle_l = shuffle_kids(n_lower_sideband,7) + 101

# shuffle_u_i = shuffle_kids(n_upper_sideband,7)
# fr_upper = np.linspace(f_lo+f_lo_gap/2,fr_max,n_upper_sideband)[shuffle_u_i]
# shuffle_u = shuffle_kids(n_upper_sideband,7) + n_lower_sideband + 101

# shuffle = np.concatenate((shuffle_l, shuffle_u))
# fr = np.concatenate((fr_lower,fr_upper))





# Hybrid section
l_hybrid = l_hybrid_min * (fr_max_at_lhybrid_min / fr)
l_hybrid_shorted = fraction_l_hybrid_shorted * l_hybrid
l_hybrid_other_one = (1 - fraction_l_hybrid_shorted) * l_hybrid

ABCD_hybrid_shorted = TL_Hybrid.ABCD(fr,l_hybrid_shorted)
ABCD_hybrid_other_one = TL_Hybrid.ABCD(fr,l_hybrid_other_one)


# MS section
l_coup2 = filterbank_geometry[:,3] * 1e-6
sep_lambda4 = filterbank_geometry[:,5] * 1e-6
l_meander = filterbank_geometry[:,6] * 1e-6


l_along_filter = l_coup2 + sep_lambda4 + 2*l_meander + width_MS_RES + width_MS_MKID*2
l_corner = 1e-6 + width_MS_MKID + 5e-6 + width_MS_MKID + 1e-6
l_to_mkid_left = (l_coup2 - width_MS_MKID) / 2 + width_MS_MKID + 5e-6
l_to_mkid_right = width_MS_RES + l_to_mkid_left

l_ms  = l_to_mkid_left + l_corner + l_along_filter + l_corner + l_to_mkid_right
ABCD_ms = TL_Microstrip.ABCD(fr,l_ms)


# Coupler

Coupler_mkid = Coupler(fr,kid_Qc,[Z0_thru_RO_line/2,TL_NbTiN.Z0],res_length='quarterwave')
ABCD_coupler = Coupler_mkid.ABCD(fr)

# %%

def find_zerointercept(x,y):
    """
        Find first zero-intercept of a line
    """
    
    i_min = np.argmin(np.abs(y))

    if i_min == (len(y) - 1):
        raise ValueError("Given range does not intercept with zero")

    x1 = x[i_min]
    y1 = y[i_min]
    x2 = x[i_min + 1]
    y2 = y[i_min + 1]

    dy = y2 - y1
    dx = x2 - x1
    slope = dy/dx
    return - y1 / slope + x1


l_wide_nbtin = np.zeros_like(l_hybrid)
l_wide_estimate = (TL_NbTiN.wavelength(fr) / 4) - l_hybrid * np.sqrt(TL_Hybrid.eps_eff / TL_NbTiN.eps_eff) - l_ms * np.sqrt(TL_Microstrip.eps_eff / TL_NbTiN.eps_eff)
n_points_to_search = int(1e3)

for i,fr_i in enumerate(fr):
    l_wide_search_range = np.linspace(0.95*l_wide_estimate[i],1.05*l_wide_estimate[i],n_points_to_search)
    Z_in_search_range = np.zeros(n_points_to_search,dtype=complex)

    for j,l_wide_guess in enumerate(l_wide_search_range):
        ABCD_wide_nbtin = TL_NbTiN.ABCD(fr_i,l_wide_guess)
        ABCD_guess = chain(ABCD_coupler[:,:,i],ABCD_wide_nbtin,ABCD_hybrid_other_one[:,:,i],ABCD_ms[:,:,i],ABCD_hybrid_shorted[:,:,i])
        Z_in_search_range[j] = Zin_from_abcd(ABCD_guess, Z_L=0)

    try:
        l_wide_nbtin[i] = find_zerointercept(l_wide_search_range,np.imag(Z_in_search_range))
    except ValueError:
        print("Line width length range is insufficient for finding correct resonance condition")
    
    



# %%

# SHOW THE RESULTING KIDS

nf = int(1e6)
f = np.linspace(4e9,6e9,nf)
S = np.zeros((2,2,nf,n_filters))
ABCD_FB = np.moveaxis(np.tile(np.eye(2),(nf,1,1)),0,-1)

TL_Readout = TransmissionLine(Z0=63.8,eps_eff=10.5)
ABCD_ROsection = TL_Readout.ABCD(f,100e-6) # 100 micron readout line between each kid

for i,fr_i in enumerate(fr):
    # Hybrid sections
    ABCD_hybrid_shorted = TL_Hybrid.ABCD(f,l_hybrid_shorted[i])
    ABCD_hybrid_other_one = TL_Hybrid.ABCD(f,l_hybrid_other_one[i])

    # Microstrip section
    ABCD_ms = TL_Microstrip.ABCD(f,l_ms[i])

    # Wide section
    ABCD_wide_nbtin = TL_NbTiN.ABCD(f,l_wide_nbtin[i])

    # Coupler
    Coupler_mkid = Coupler(fr_i,kid_Qc,[Z0_thru_RO_line/2,TL_NbTiN.Z0],res_length='quarterwave')
    ABCD_coupler = Coupler_mkid.ABCD(f)

    ABCD_MKID = chain(ABCD_coupler,ABCD_wide_nbtin,ABCD_hybrid_other_one,ABCD_ms,ABCD_hybrid_shorted)
    Z_MKID = Zin_from_abcd(ABCD_MKID,Z_L=0)

    ABCD_MKID_shunt_in_RO = abcd_shuntload(Z_MKID)

    ABCD_FB = chain(ABCD_FB,ABCD_MKID_shunt_in_RO,ABCD_ROsection)

# %%

# plot data

S_FB = abcd2s(ABCD_FB,Z0=Z0_thru_RO_line)
S21_absSq = 20 * np.log10(np.abs(S_FB[1,0,:]))

fig, ax = plt.subplots(figsize=(20, 20), layout='constrained')

ax.plot(f/1e9,S21_absSq)

fig.show()


# %% 

# MAKE THE MASTER KIDLIST FILE


# Get KID coupler geometry data corresponding to Qc desired

kid_coupler_data = np.genfromtxt("kid_coupler_data.csv",delimiter=',',skip_header=1)

f_kid_coupler = kid_coupler_data[:,0]
l_coup = kid_coupler_data[:,1]

kid_coupler_fit = np.polyfit(1/f_kid_coupler,l_coup,3)

f_kid_coupler_x = np.linspace(3,7,int(1e4))
kid_coupler_interp = np.polyval(kid_coupler_fit,1/f_kid_coupler_x)

fig, ax = plt.subplots(figsize=(20, 20), layout='constrained')

ax.plot(f_kid_coupler,l_coup,linestyle='none',marker='x',markersize=4)
ax.plot(f_kid_coupler_x,kid_coupler_interp)

fig.show()


# %%
# master_kidlist columns:
# KID ID,type,f_KID_design[GHz],F_filter_design [GHz],KID Qc,l_al[mm],l_wide[mm],l_thz[um],l_coupler[um]

## Manual addition of wb kids, blind kids and NbTiN kids

wb_kids = pd.DataFrame(
    {"KID ID": [0,1,2,3], 
     "type": ['wideband_bf','wideband_bf','wideband_af','wideband_af'], 
     "f_KID_design [GHz]": [5.9,5.91,5.93,5.94], 
     "F_filter_design [GHz]": np.nan, 
     "KID Qc": 20e3,
     "l_al[mm]": [1.220,1.218,1.214,1.212],
     "l_wide[mm]": [2.919,2.914,2.904,2.899],
     "l_thz[um]": 40.4,
     "l_coupler[um]": [39.991,39.919,39.777,39.707]
    }
)


blind_kids = pd.DataFrame(
    {"KID ID": [10,11,12,13], 
     "type": ['blind_perp','blind_perp','blind_para','blind_para'], 
     "f_KID_design [GHz]": [4.1,4.105,4.115,4.12], 
     "F_filter_design [GHz]": np.nan, 
     "KID Qc": 50e3,
     "l_al[mm]": [1.756,1.754,1.750,1.748],
     "l_wide[mm]": [4.285,4.280,4.269,4.264],
     "l_thz[um]": 0,
     "l_coupler[um]": [36.196,36.149,36.056,36.009]
    }
)


nbtin_kids = pd.DataFrame(
    {"KID ID": [30,31], 
     "type": 'NbTiN_cpw', 
     "f_KID_design [GHz]": [6.1, 6.11], 
     "F_filter_design [GHz]": np.nan, 
     "KID Qc": 2e5,
     "l_al[mm]": 0,
     "l_wide[mm]": [3.978,3.971],
     "l_thz[um]": 0,
     "l_coupler[um]": [10.717,10.695]
    }
)


# Filter row format:
# index,f0 [GHz],l_coup1 [um],l_coup2 [um],l_res [um],sep_lambda4 [um],l_meander [um]

# master_kidlist columns:
# KID ID,type,f_KID_design[GHz],F_filter_design [GHz],KID Qc,l_al[mm],l_wide[mm],l_thz[um],l_coupler[um]

filter_kids_df = pd.DataFrame(
    {"KID ID": shuffle, 
     "type": 'filter', 
     "f_KID_design [GHz]": fr/1e9, 
     "F_filter_design [GHz]": filterbank_geometry[:,1], 
     "KID Qc": kid_Qc,
     "l_al[mm]": l_hybrid*1e3,
     "l_wide[mm]": l_wide_nbtin*1e3,
     "l_thz[um]": l_ms*1e6,
     "l_coupler[um]": np.polyval(kid_coupler_fit,1/(fr/1e9))
    }
)



master_kidlist = pd.concat((filter_kids_df,wb_kids,blind_kids,nbtin_kids)).sort_values(by='KID ID')

np.savetxt("master_kidlist_D2.1.csv",master_kidlist,delimiter=',',fmt="%s",header="KID ID,type,f_KID_design[GHz],F_filter_design [GHz],KID Qc,l_al[mm],l_wide[mm],l_thz[um],l_coupler[um]")


# %%