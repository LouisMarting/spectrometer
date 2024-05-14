#%%
## Make a kidlist based on input from the filterbank_geometry.csv
## Author: Louis Marting
## Date: December 2023

# Importing packages
import json
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from kidencoding import shuffle_kids, shuffle_kids2

shuffle_old = shuffle_kids(197,7)
# shuffle = shuffle_kids2(197,np.array([[12,4],[12,4],[8,7],[9,5]]));
# shuffle = shuffle_kids2(197,np.array([[10,7],[7,9],[8,8]]));
shuffle = shuffle_kids2(200,np.array([[10,10],[10,10]]));

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(shuffle,marker="x",markersize="4")
ax.plot(shuffle_old,marker="x",markersize="4")
ax.set_xlabel("chip position")
ax.set_ylabel("fr")

shuffle = shuffle_kids2(197,np.array([[10,7],[7,9],[8,8]]));

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(shuffle,marker="x",markersize="4")
ax.plot(shuffle_old,marker="x",markersize="4")
ax.set_xlabel("chip position")
ax.set_ylabel("fr")

shuffle = shuffle_kids2(197,np.array([[12,4],[12,4],[8,7],[9,5]]));

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(shuffle,marker="x",markersize="4")
ax.plot(shuffle_old,marker="x",markersize="4")
ax.set_xlabel("chip position")
ax.set_ylabel("fr")


# %%
# Investigating the f-scatter across the filterbank


KID_data = np.load("FitSweep_fit.npy")

# KID data columns: [kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi]

kidid = KID_data[0,:]
fr = KID_data[1,:]
dfr = KID_data[2,:]
Qr = KID_data[3,:]
dQr = KID_data[4,:]
Qc = KID_data[5,:]
dQc = KID_data[6,:]
Qi = KID_data[7,:]
dQi = KID_data[8,:]


## KIDLIST DATA
master_kidlist_data = np.genfromtxt("masterlist_220-440GHzFullDESHIMA2.0.csv",comments='#',delimiter=",")


masterkid_id = master_kidlist_data[:,0]
fr_design = master_kidlist_data[:,2]
f0_design = master_kidlist_data[:,3]

# remove KIDS that are not part of filterbank
masterkid_id = masterkid_id[~np.isnan(f0_design)]
fr_design = fr_design[~np.isnan(f0_design)]
f0_design = f0_design[~np.isnan(f0_design)] # do last

masterkid_id = masterkid_id[np.flip(np.argsort(f0_design))]
fr_design = fr_design[np.flip(np.argsort(f0_design))]
f0_design = f0_design[np.flip(np.argsort(f0_design))] # do last


# MEASUREMENT DATA
with open('kid_corresp.json') as f:
   kid_correspondence = json.load(f)

kids_found_id = [int(k) for k in list(kid_correspondence.keys())]
masterkid_id_found = masterkid_id[np.isin(masterkid_id,kids_found_id)]
masterkid_id_found = np.array([int(k) for k in masterkid_id_found]) #id of design
fr_design_found = fr_design[np.isin(masterkid_id,kids_found_id)]

# physical position ordered index array
kidid_pos_ordered = np.array([kid_correspondence.get(str(key)) for key in masterkid_id_found])


kidid_order_l = np.array([kid_correspondence.get(str(key)) for key in masterkid_id_found[masterkid_id_found < 274]])
kidid_order_u = np.array([kid_correspondence.get(str(key)) for key in masterkid_id_found[masterkid_id_found >= 274]])


fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found < 274],fr[kidid_order_l])
ax.scatter(fr_design_found[masterkid_id_found >= 274],fr[kidid_order_u])
ax.scatter(fr_design_found[masterkid_id_found < 274],fr_design_found[masterkid_id_found < 274])
ax.scatter(fr_design_found[masterkid_id_found >= 274],fr_design_found[masterkid_id_found >= 274])
ax.set_title("MKID frequency")
ax.set_ylabel("fr [GHz]")
ax.set_xlabel("design fr [GHz]")

dfr_l = fr[kidid_order_l] - fr_design_found[masterkid_id_found < 274]
dfr_u = fr[kidid_order_u] - fr_design_found[masterkid_id_found >= 274]

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(dfr_l,linestyle='none',marker="x",markersize="4")
ax.plot(dfr_u,linestyle='none',marker="x",markersize="4")
ax.set_title("MKID frequency")
ax.set_ylabel("fr - design fr [GHz]")
ax.set_xlabel("chip order")

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found < 274],dfr_l)
ax.scatter(fr_design_found[masterkid_id_found >= 274],dfr_u)
ax.set_title("MKID frequency")
ax.set_ylabel("fr - design fr [GHz]")
ax.set_xlabel("design fr [GHz]")

# %%
def linear_fit(x,y):
    # y = m*x + b
    m, b = np.polyfit(x, y, deg=1)
    return m, b

def array_of_linfit(m,b,x):
    # y = m*x + b
    return m*x + b


# fit lower sideband 

m, b = linear_fit(fr_design_found[masterkid_id_found < 274],dfr_l)

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found < 274],dfr_l)
ax.axline(xy1=(0, b), slope=m, label=f'$y = {m:.1f}x {b:+.1f}$')
ax.set_xlim(4,5)
ax.set_ylim(-0.6,-0.4)

dfr_l_offset = array_of_linfit(m,b,fr_design_found[masterkid_id_found < 274])
dfr_l_residue = dfr_l - dfr_l_offset

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found < 274],dfr_l_residue*1e3)
ax.set_title("MKID frequency")
ax.set_ylabel("dfr [MHz]")
ax.set_xlabel("design fr [GHz]")

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(dfr_l_residue*1e3,linestyle='none',marker="x",markersize="4")
ax.set_title("MKID frequency")
ax.set_ylabel("dfr [MHz]")
ax.set_xlabel("chip order")

print("lower sideband")
print(f"std: {np.std(dfr_l_residue*1e3):.2f} MHz\n")


# fit upper sideband 

m, b = linear_fit(fr_design_found[masterkid_id_found >= 274],dfr_u)

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found >= 274],dfr_u)
ax.axline(xy1=(0, b), slope=m, label=f'$y = {m:.1f}x {b:+.1f}$')
ax.set_xlim(5,6)
ax.set_ylim(-0.7,-0.5)

dfr_u_offset = array_of_linfit(m,b,fr_design_found[masterkid_id_found >= 274])
dfr_u_residue = dfr_u - dfr_u_offset

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found >= 274],dfr_u_residue*1e3)
ax.set_title("MKID frequency")
ax.set_ylabel("dfr [MHz]")
ax.set_xlabel("design fr [GHz]")

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(dfr_u_residue*1e3,linestyle='none',marker="x",markersize="4")
ax.set_title("MKID frequency")
ax.set_ylabel("dfr [MHz]")
ax.set_xlabel("chip order")

print("upper sideband")
print(f"std: {np.std(dfr_u_residue*1e3):.2f} MHz\n")

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found < 274],dfr_l_residue*1e3)
ax.scatter(fr_design_found[masterkid_id_found >= 274],dfr_u_residue*1e3)
ax.set_title("MKID frequency")
ax.set_ylabel("dfr [MHz]")
ax.set_xlabel("design fr [GHz]")

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(dfr_l_residue*1e3,linestyle='none',marker="x",markersize="4")
ax.plot(dfr_u_residue*1e3,linestyle='none',marker="x",markersize="4")
ax.set_title("MKID frequency")
ax.set_ylabel("dfr [MHz]")
ax.set_xlabel("chip order")

# %% 
# Plot per nth in staircase pattern

fr_design_l = fr_design_found[masterkid_id_found < 274]
fr_design_u = fr_design_found[masterkid_id_found >= 274]
masterkid_id_found_l = masterkid_id_found[masterkid_id_found < 274]
masterkid_id_found_u = masterkid_id_found[masterkid_id_found >= 274]


group_size = 25

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
for i in np.arange(100,274,group_size):
    range = np.arange(i,i+group_size)
    idx_found = np.isin(masterkid_id_found_l,range)
    
    ax.scatter(masterkid_id_found_l[idx_found],dfr_l_residue[idx_found]*1e3)


# %%
# Plot Qc

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found < 274],Qc[kidid_order_l])
ax.scatter(fr_design_found[masterkid_id_found >= 274],Qc[kidid_order_u])
ax.set_title("MKID frequency")
ax.set_ylabel("Qc")
ax.set_xlabel("design fr [GHz]")
ax.set_yscale('log')

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(Qc[kidid_order_l],linestyle='none',marker="x",markersize="4")
ax.plot(Qc[kidid_order_u],linestyle='none',marker="x",markersize="4")
ax.set_title("MKID frequency")
ax.set_ylabel("Qc")
ax.set_xlabel("chip order")
ax.set_yscale('log')

# %%
# Plot Qi

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.scatter(fr_design_found[masterkid_id_found < 274],Qi[kidid_order_l])
ax.scatter(fr_design_found[masterkid_id_found >= 274],Qi[kidid_order_u])
ax.set_title("MKID frequency")
ax.set_ylabel("Qi")
ax.set_xlabel("design fr [GHz]")
ax.set_yscale('log')

fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
ax.plot(Qi[kidid_order_l],linestyle='none',marker="x",markersize="4")
ax.plot(Qi[kidid_order_u],linestyle='none',marker="x",markersize="4")
ax.set_title("MKID frequency")
ax.set_ylabel("Qi")
ax.set_xlabel("chip order")
ax.set_yscale('log')


# %% Apply the new shuffle to measured data.

plt.show()