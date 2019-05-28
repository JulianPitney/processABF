# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:10:12 2019

@author: User1
"""

from scipy.signal import iirfilter, lfilter, hilbert, find_peaks, detrend
from neo import AxonIO
import numpy as np
import matplotlib.pyplot as plt


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
import scipy.optimize as sp_opt

def bandpass_filter(data, cutoff, fs):
    order = 3
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype='band',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data    

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def exp_decay_inc(t, a, b, k):
    y = b + a*(1-np.exp(-t*k))  
    return y  

def f_filter(data, cutoff, fs,type_f):
    order = 3
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype=type_f,
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def get_Rsquared(data,yEXP):
  residuals = data-yEXP
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((data-np.mean(data))**2)
  r_squared = 1 - (ss_res / ss_tot)
  return r_squared
   
def fit_exp(trace,dt,fit_inc,min_fit):
    len_trace = len(trace)
    t = np.arange(len_trace)
    nb_fits = int(np.floor(len_trace/fit_inc))
    R_list = np.zeros(nb_fits)
    tau_list = np.zeros(nb_fits)
    yEXP_list = []
    for i in range(nb_fits):
        sig = trace[:min_fit+(fit_inc*i)]
        try:
            b = np.mean(sig[:5])
            asymptote =0 #np.mean(sig[-5:])
            params = [asymptote-b,b,0.01]
            [popt, pcov] = sp_opt.curve_fit(exp_decay_inc, t[:min_fit+(fit_inc*i)], sig,p0=params,maxfev=5000)#
            yEXP = exp_decay_inc(t[:min_fit+(fit_inc*i)], *popt)
            yEXP_list.append(yEXP)
            if popt[2] > 0:
                tau_list[i] = (1/popt[2])*dt
                R_list[i] = get_Rsquared(sig,yEXP)
            else:
                tau_list[i] = np.nan
                R_list[i] = np.nan
        except RuntimeError:
            tau_list[i] = np.nan
            R_list[i] = np.nan
            yEXP_list.append(np.nan)
    min_R = 0.8
    R_list[np.where(R_list<min_R)] = np.nan
    if np.isnan(np.nanmax(R_list)):
        tau = np.nan
        R = np.nan
        yEXP = np.nan
    else:
        tau = tau_list[np.where(~np.isnan(R_list))[0][-1]]
        R = R_list[np.where(~np.isnan(R_list))[0][-1]]
        yEXP = yEXP_list[np.where(~np.isnan(R_list))[0][-1]]
    return tau,R,yEXP



#Varisables to specify       
folder_path = "C:/Projects/processABF/Kieran_Data/"
ABF_name = "19507004.abf"
signal_type = 0       #Current:0, Voltage:1 (ou inverse)


file_name = ABF_name.split('.')[0]
ABF_path = folder_path + ABF_name
original_file = AxonIO(filename=ABF_path)
read_data = original_file.read_block(lazy=False)  
nb_steps = len(read_data.segments[0].analogsignals[0]) 
nb_sweeps = len(read_data.segments)
fs = np.array(read_data.segments[0].analogsignals[0].sampling_rate,dtype=int)
dt = 1/fs
traces = np.zeros((nb_steps,nb_sweeps))


include_sweeps = np.arange(nb_sweeps) #For all sweeps
#OR specify manually (index starts at 0) e.g.
#include_sweeps = [0,1,4,6]

lp_cutoff = 0.01 #Low pass cutoff
for sw_i in range(nb_sweeps):
  traces[:,sw_i] = np.ravel(np.array(read_data.segments[sw_i].analogsignals[signal_type]))
  raw = np.ravel(np.array(read_data.segments[sw_i].analogsignals[signal_type]))
  traces[:,sw_i] = detrend(traces[:,sw_i])
  baseline = traces[:,sw_i].copy()
  baseline[np.where(baseline<0)[0]] = np.nan
  traces[:,sw_i] = traces[:,sw_i] - np.nanmean(baseline)

 
cutoff = [75,275]    #75,250
  
sweep_i = 0
start = 0
start_t = int(start/dt)
sweep = traces[start_t:,sweep_i]  
nb_steps = len(sweep)
len_sweep = nb_steps*dt
f_sweep = bandpass_filter(sweep, cutoff, fs)
h_sweep = hilbert(f_sweep)
amplitude_envelope = np.abs(h_sweep)
window_env = 21
amplitude_envelope = smooth(amplitude_envelope,window_len=window_env)[int((window_env-1)/2):-int((window_env-1)/2)]




threshold = 4#3 #2
peaks,_ = find_peaks(amplitude_envelope, height=threshold, prominence=(1.5, None)) #(1, None)

gap = 400
peaks = peaks[np.where(peaks>=gap)[0]]

window_pre = 40
window_post = 25
total_window = window_pre+ window_post
nb_peaks = len(peaks)
peaks_t = np.zeros(nb_peaks)
peaks_v = np.zeros(nb_peaks)
peaks_f = np.zeros(nb_peaks)
window_len = 20
smooth_sweep = smooth(sweep,window_len=window_len)[int((window_len-1)/2):-int((window_len-1)/2)]
for i in range(nb_peaks):
    peaks_t[i] = peaks[i] - (window_pre - np.argmin(smooth_sweep[peaks[i]-window_pre:peaks[i]+window_post]))
    peaks_v[i] = np.min(smooth_sweep[peaks[i]-window_pre:peaks[i]+window_post])
    peaks_f[i] = amplitude_envelope[peaks[i]]
    
min_diff = 100    
ISI = np.diff(peaks_t)
comb_peaks = np.where(ISI<=min_diff)[0]
del_items = []
for i in comb_peaks:
    v1 = peaks_v[i]
    v2 = peaks_v[i]
    if v1<v2:
        del_items.append(i+1)
    else:
        del_items.append(i)

peaks_t = np.delete(peaks_t,del_items)
peaks_v = np.delete(peaks_v,del_items)
peaks_t = peaks_t.astype(np.int32)
nb_peaks = len(peaks_t)

#Fit decay
min_decay_t = int(0.03/dt)
max_decay_t = int(0.1/dt)
fit_inc = int(0.002/dt)
peaks_tau = np.full(nb_peaks,np.nan)
peaks_decay = []
for i in range(nb_peaks):
    print(i)
    if peaks_t[i]+max_decay_t < nb_steps:
        tau,R,yEXP = fit_exp(sweep[peaks_t[i]:peaks_t[i]+max_decay_t],dt,fit_inc,min_decay_t)
        peaks_tau[i] = tau
        peaks_decay.append(yEXP)
        
tau_times = []
tau_vals = []
dt_plot = 20
peaks_with_fits = np.where(~np.isnan(peaks_tau))[0]
nb_peaks_with_fits = len(peaks_with_fits)
for i in range(nb_peaks_with_fits):
    peak_t = peaks_t[peaks_with_fits[i]]
    yEXP = peaks_decay[peaks_with_fits[i]]
    len_fit = len(yEXP)
    tau_times.append(np.arange(peak_t,peak_t+len_fit)[np.arange(0,len(yEXP),dt_plot)])
    tau_vals.append(yEXP[np.arange(0,len(yEXP),dt_plot)])
    
tau_times = np.concatenate( tau_times, axis=0 )
tau_vals =  np.concatenate( tau_vals, axis=0 )

data = np.array([peaks_t*dt,peaks_v,peaks_tau]).T
with open(folder_path+ABF_name.split('.')[0] + '.csv','w') as f:
    np.savetxt(f,data,delimiter = ',',header = "Time (s),Amplitude,Tau (s)")
    
#Generate plot 

xaxis =  np.linspace(0,len_sweep,nb_steps)

mark0 = go.Scatter(
    x = xaxis[peaks_t],
    y = peaks_v,
    mode = 'markers',
    name = 'peaks',
    marker = dict(
        size = 7,
        color = 'rgb(255, 50, 50)',
        line = dict(
            width = 2,
    ))
)
trace0 = go.Scatter(
    x = xaxis,
    y = smooth_sweep,
    name = 'Sweep',
    line = dict(
        color = ('rgb(50, 170, 50)'),
        width = 4,)
)
trace1 = go.Scatter(
    x = xaxis,
    y = amplitude_envelope,
    mode = 'lines',
    name = 'Filter'
)
mark1 = go.Scatter(
    x = xaxis[peaks],
    y = peaks_f,
    mode = 'markers',
    name = 'peaks'
)

taus = go.Scatter(
    x = xaxis[tau_times],
    y = tau_vals,
    mode = 'markers',
    name = 'Taus',
    marker = dict(
        size = 5,
        color = 'rgb(0,0, 255)',
        )
)

data = [trace0,mark0,taus,trace1,mark1]

# =============================================================================
# layout = go.Layout(
#     title='Spontaneous events detection',
#     yaxis=dict(
#         title=''
#     ),
#     yaxis2=dict(
#         title='',
#         titlefont=dict(
#             color='rgb(148, 103, 189)'
#         ),
#         tickfont=dict(
#             color='rgb(148, 103, 189)'
#         ),
#         overlaying='y',
#         side='right'
#     ),
#     yaxis3=dict(
#         title='',
#         titlefont=dict(
#             color='rgb(148, 103, 189)'
#         ),
#         tickfont=dict(
#             color='rgb(255, 0, 0)'
#         ),
#         overlaying='y',
#         side='right'
#     )
# )
# =============================================================================
fig = go.Figure(data=data)
plot(fig, auto_open=True)
#py.iplot(fig, filename='multiple-axes-double')