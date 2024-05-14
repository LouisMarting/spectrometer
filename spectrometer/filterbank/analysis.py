import numpy as np
import numpy.ma as ma
from scipy.signal import savgol_filter,find_peaks
import matplotlib as mpl
from matplotlib import pyplot as plt
import time

from .components import Filterbank


def analyse():
    pass

def analyse_variance(Filterbank : Filterbank,f,n_filterbanks=1):
    f0 = Filterbank.f0
    Ql = 100   #### HARDCODED, should be changed later

    f0_realized = [[]] * n_filterbanks
    Ql_realized = [[]] * n_filterbanks
    avg_envelope_eff = np.empty(n_filterbanks)
    usable_spectral_fraction = np.empty(n_filterbanks)
    for i in range(n_filterbanks):
        Filterbank.reset_and_shuffle()
        Filterbank.S(f)

        f0_realized[i],Ql_realized[i],_,_ = Filterbank.realized_parameters()

        avg_envelope_eff[i], usable_spectral_fraction[i] = filterbank_analysis(Filterbank,f0_realized[i],Ql_realized[i])




    f0_realized = np.array(f0_realized).ravel()
    Ql_realized = np.array(Ql_realized).ravel()
    
    df_variance = Ql * (f0_realized - np.tile(f0,n_filterbanks)) / np.tile(f0,n_filterbanks)
    Ql_variance = (Ql_realized-Ql) / Ql
    
    return f0_realized, Ql_realized, df_variance, Ql_variance, avg_envelope_eff, usable_spectral_fraction
    
def filterbank_analysis(Filterbank : Filterbank,f0_realized,Ql_realized):
    f = Filterbank.f
    envelope = np.array(Filterbank.S31_absSq_list).max(axis=0)
    fb_f_min = f0_realized[-1] - (f0_realized[-1]/Ql_realized[-1])/2
    fb_f_max = f0_realized[0] + (f0_realized[0]/Ql_realized[0])/2

    inband = np.logical_and(f > fb_f_min,f < fb_f_max)

    # Average envelope efficiency
    avg_envelope_eff = ma.masked_array(envelope,mask=~inband).mean()

    # Usable spectral fraction
    responsive_filter = np.logical_and(inband,envelope > (avg_envelope_eff / np.sqrt(2)))

    usable_spectral_faction = np.sum(responsive_filter)/np.sum(inband)

    return avg_envelope_eff, usable_spectral_faction


colors = [
    np.array([90,136,237])/255,
    np.array([86,199,74])/255,
    np.array([242,131,45])/255,
    np.array([132,82,201])/255,
    np.array([108,98,96])/255
]

def plot_measurements(data):
    window_size = 17

    f = data[:,0]

    mask = f == 283.7

    i_ray = np.argmax(mask)

    data[i_ray,1:] = (data[i_ray-1,1:] + data[i_ray+1,1:])/2

    filter = data[:,1]

    wb1 = data[:,2]

    wb2 = data[:,3]

    fig, (ax1, ax2, ax3) =plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(5.5,3.5),layout='constrained')

    ax2.plot(f,filter,label='filter',color=colors[2])

    ax1.plot(f,wb1,label='wideband1',color=colors[0],alpha=0.4)
    ax1.plot(f,wb2,label='wideband2',color=colors[1],alpha=0.4)
    ax3.plot(f,filter/wb1, label='filter/wideband1',color=colors[3])

    
    ax1.set_title("Raw Measurement Data")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    # ax.set_yscale("log")
    # ax.set_ylim()
    ax3.set_xlabel('Frequency [GHz]')
    ax1.set_xlim(np.min(f),np.max(f))
    # axs[0].set_xlim(200,250)

    fig.supylabel('Response [a.u.]')


def plot_measurements_filter(data, data2=None):
    window_size = 17

    f = data[:,0]

    #remove a single outlier
    mask = f == 283.7

    i_ray = np.argmax(mask)

    data[i_ray,1:] = (data[i_ray-1,1:] + data[i_ray+1,1:])/2
    filter = data[:,1]
    wb1 = data[:,2]
    wb2 = data[:,3]

    filter_div_wb1_smooth = savgol_filter(filter/wb1,window_size,2)
    filter_div_wb2_smooth = savgol_filter(filter/wb2,window_size,2)
    max_data1 = np.max(filter_div_wb1_smooth)

    ############ fit Q, f0 ########
    n_interp = 20
    f0 = 227
    Ql = 100

    filt_smooth = filter_div_wb1_smooth

    fq = np.linspace(f[0],f[-1],n_interp*len(f))
 
    filt_smooth_q = np.interp(fq,f,filt_smooth)

    i_peaks,peak_properties = find_peaks(filt_smooth_q,height=0.5*max(filt_smooth_q),prominence=0.05)

    i_peak = i_peaks[np.argmax(peak_properties["peak_heights"])]

    # f0, as realized in the filter
    f0_realized = fq[i_peak]

    # Find FWHM manually:
    HalfMaximum = filt_smooth_q[i_peak] / 2
    diff_from_HalfMaximum = np.abs(filt_smooth_q-HalfMaximum)

    # search window = +/- a number of filter widths
    search_range = [f0_realized-3*f0/Ql, f0_realized+3*f0/Ql]
    
    search_window = np.logical_and(fq > search_range[0],fq < f0_realized)
    i_HalfMaximum_lower = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    search_window = np.logical_and(fq > f0_realized,fq < search_range[-1])
    i_HalfMaximum_higher = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    fwhm = fq[i_HalfMaximum_higher] - fq[i_HalfMaximum_lower]

    Ql_realized = f0_realized / fwhm

    print(f"f0 realized: {f0_realized} GHz")
    print(f"Ql realized: {Ql_realized}")


    fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')

    ax.plot(f,filter_div_wb1_smooth,label='filter/wb1',color=colors[0])
    ax.plot(f,filter_div_wb2_smooth,label='filter/wb2',color=colors[1])
    
    if data2 is not None:
        data2[i_ray,1:] = (data2[i_ray-1,1:] + data2[i_ray+1,1:])/2
        filter = data2[:,1]
        wb1 = data2[:,2]
        wb2 = data2[:,3]

        filter_div_wb1_smooth = savgol_filter(filter/wb1,window_size,2)
        filter_div_wb2_smooth = savgol_filter(filter/wb2,window_size,2)

        scaling = max_data1/np.max(filter_div_wb1_smooth)*0.4

        ax.plot(f,filter_div_wb1_smooth*scaling,label='filter/wb1\n(overdriven)',color=colors[0],alpha=0.4,linestyle="--")
        ax.plot(f,filter_div_wb2_smooth*scaling,label='filter/wb2\n(overdriven)',color=colors[1],alpha=0.4,linestyle="--")
    
    ax.set_title("Filter Response")
    # ax.set_yscale("log")
    ax.set_xlabel('Frequency [GHz]')
    ax.legend()

    ax.set_xlim(220,234)

    fig.supylabel('Response [a.u.]')

def plot_measurements_presentation(data):
    window_size = 17

    f = data[:,0]

    mask = f == 283.7

    i_ray = np.argmax(mask)

    data[i_ray,1:] = (data[i_ray-1,1:] + data[i_ray+1,1:])/2

    filter = data[:,1]

    wb1 = data[:,2]

    wb2 = data[:,3]

    fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(6,3),layout='constrained')


    ax.plot(f,filter/wb1, label='filter/wideband1',color=colors[3])

    
    ax.set_title("Measured Filter Response")

    # ax.set_yscale("log")
    # ax.set_ylim()
    ax.set_xlabel('Frequency [GHz]')
    ax.set_xlim(np.min(f),np.max(f))
    # axs[0].set_xlim(200,250)

    ax.set_ylabel('Response [a.u.]')

    
