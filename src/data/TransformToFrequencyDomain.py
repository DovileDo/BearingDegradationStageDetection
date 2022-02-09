import numpy as np
import pandas as pd
import time
from scipy.fft import rfft, rfftfreq
from scipy import signal


def ToFrequency(df):
    X = np.empty((0,1282), float)
    
    for i in range(0,len(df),2560):
        Hshort = signal.decimate(df['Hacc'][i:i+2560],2)
        Vshort = signal.decimate(df['Vacc'][i:i+2560],2)
        Hf = np.abs(rfft(Hshort))
        Vf = np.abs(rfft(Vshort))
        x = np.append(Hf,Vf)
        X = np.append(X, np.array([x]), axis=0)
        
    return X