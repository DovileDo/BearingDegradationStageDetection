import numpy as np
import pandas as pd
import os
from scipy.stats import kurtosis
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy import signal

def ToTime(df):

    Hzerocross = []
    Vzerocross = []
    Hkurtosis = []
    Vkurtosis = []
    Hrms = []
    Vrms = []
    Hpeaks = []
    Vpeaks = []
    Hmean = []
    Vmean = []
    Hstd = []
    Vstd = []
    Hmedian = []
    Vmedian = []
    Hskewness = []
    Vskewness = []
    Henergy = []
    Venergy = []
    Hshapiro = []
    Vshapiro = []
    Hkl = []
    Vkl = []
    Hrkl = []
    Vrkl = []
    Hcrest = []
    Vcrest = []

    for i in range(0,len(df),2560):
        Horizontal = signal.decimate(df['Hacc'][i:i+2560],2)
        Vertical = signal.decimate(df['Vacc'][i:i+2560],2)
        
        #zero crossing
        Hc = ((Horizontal[:-1] * Horizontal[1:]) < 0).sum()+ (Horizontal == 0).sum()
        Hzerocross.append(Hc)
        Vc = ((Vertical[:-1] * Vertical[1:]) < 0).sum()+ (Vertical == 0).sum()
        Vzerocross.append(Vc)
        
        #kurtosis: distribution tail
        Hkur = kurtosis(Horizontal)
        Hkurtosis.append(Hkur)
        Vkur = kurtosis(Vertical)
        Vkurtosis.append(Vkur)
        
        #RMS  root-mean-square value of a signal: Loosely speaking, it represents the average "power" of a signal.
        #The root mean square (RMS) is widely used to represent HIs and is further employed to identify the initial
        #point of degradation when the RMS exceeds a pre-specified threshold
        # Wang, W. A model to predict the residual life of rolling element bearings given monitored condition information
        #to date. IMA J. Manag. Math. 2002, 13, 3–16.
        hrms = np.sqrt(np.mean(Horizontal**2))
        Hrms.append(hrms)
        vrms = np.sqrt(np.mean(Vertical**2))
        Vrms.append(vrms)
        
        #Number of peaks
        hpeaks, _ = find_peaks(Horizontal)
        Hpeaks.append(len(hpeaks)/len(Horizontal))
        vpeaks, _ = find_peaks(Vertical)
        Vpeaks.append(len(vpeaks)/len(Vertical))

        #Mean
        Hmean.append(np.mean(Horizontal))
        Vmean.append(np.mean(Vertical))
        
        #Median
        Hmedian.append(np.median(abs(Horizontal)))
        Vmedian.append(np.median(abs(Vertical)))
        
        #std
        Hstd.append(np.std(Horizontal))
        Vstd.append(np.std(Vertical))

        #Skewness
        Hskewness.append(stats.skew(Horizontal))
        Vskewness.append(stats.skew(Vertical))
        
        #Crest
        Hcrest.append(np.max(np.abs(Horizontal))/np.sqrt(np.mean(np.square(Horizontal))))
        Vcrest.append(np.max(np.abs(Vertical))/np.sqrt(np.mean(np.square(Vertical))))
        
        #Energy
        Henergy.append(np.sum(np.abs(Horizontal)**2))
        Venergy.append(np.sum(np.abs(Vertical)**2))

        #Shapiro
        Hs, Hp = stats.shapiro(Horizontal)
        Hshapiro.append(Hs)
        Vs, Vp = stats.shapiro(Vertical)
        Vshapiro.append(Vs)
        
        #KL
        x = np.linspace(min(Horizontal), max(Horizontal), 100)
        Hen = stats.entropy(stats.gaussian_kde(Horizontal).evaluate(x), stats.norm.pdf(x, np.mean(Horizontal), np.std(Horizontal)))
        Hkl.append(Hen)
        x = np.linspace(min(Vertical), max(Vertical), 100)
        Ven = stats.entropy(stats.gaussian_kde(Vertical).evaluate(x), stats.norm.pdf(x, np.mean(Vertical), np.std(Vertical)))
        Vkl.append(Ven)
        
        #Reverse KL
        x = np.linspace(min(Horizontal), max(Horizontal), 100)
        Hren = stats.entropy(stats.norm.pdf(x, np.mean(Horizontal), np.std(Horizontal)), stats.gaussian_kde(Horizontal).evaluate(x))
        Hrkl.append(Hren)	
        x = np.linspace(min(Vertical), max(Vertical), 100)
        Vren = stats.entropy(stats.norm.pdf(x, np.mean(Vertical), np.std(Vertical)), stats.gaussian_kde(Vertical).evaluate(x))
        Vrkl.append(Vren)

    df = pd.DataFrame(Hzerocross,columns=['Hzerocross'])
    df['Hkurtosis'] = Hkurtosis
    df['Hrms'] = Hrms
    df['Hpeaks'] = Hpeaks
    df['Hmean'] = Hmean
    df['Hstd'] = Hstd
    df['Hmedian'] = Hmedian
    df['Hskewness'] = Hskewness
    df['Hcrest'] = Hcrest
    df['Henergy'] = Henergy
    df['Hshapiro'] = Hshapiro
    df['Hkl'] = Hkl
    df['Hrkl'] = Hrkl
    df['Vzerocross'] = Vzerocross
    df['Vkurtosis'] = Vkurtosis
    df['Vrms'] = Vrms
    df['Vpeaks'] = Vpeaks
    df['Vmean'] = Vmean
    df['Vstd'] = Vstd
    df['Vmedian'] = Vmedian
    df['Vskewness'] = Vskewness
    df['Vcrest'] = Vcrest
    df['Venergy'] = Venergy
    df['Vshapiro'] = Vshapiro
    df['Vkl'] = Vkl
    df['Vrkl'] = Vrkl

    mask = df['Hkl'] != np.inf
    df.loc[~mask, 'Hkl'] = df.loc[mask, 'Hkl'].max()

    mask = df['Vkl'] != np.inf
    df.loc[~mask, 'Vkl'] = df.loc[mask, 'Vkl'].max()

    mask = df['Hrkl'] != np.inf
    df.loc[~mask, 'Hrkl'] = df.loc[mask, 'Hrkl'].max()

    mask = df['Vrkl'] != np.inf
    df.loc[~mask, 'Vrkl'] = df.loc[mask, 'Vrkl'].max()

    return df