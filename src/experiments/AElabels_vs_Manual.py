import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal
import pandas as pd
import numpy as np
import os

import sys
sys.path.append('../data')

from TransformToFrequencyDomain import ToFrequency

sys.path.append('../models')

from AutoEncoder import get_AElabels


# Number of samples in normalized_tone
SAMPLE_RATE = 25600
DURATION = 0.1
N = int(SAMPLE_RATE * DURATION)

Hmeans = []
Vmeans = []
Hfrequencies = []
Vfrequencies = []

b = 'Bearing1_1'

rootdir = '../../data/raw/Train_set/' + b
for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		df = pd.read_csv(os.path.join(subdir, file),
						header=None,
						names=['h','m','s','ms','Hacc', 'Vacc'])
		Hf = np.abs(rfft(df['Hacc'].to_numpy()))
		Vf = np.abs(rfft(df['Vacc'].to_numpy()))
		xf = rfftfreq(N, 1 / SAMPLE_RATE)

		Hpositive = abs(df['Hacc'].to_numpy())
		Hmax = np.argpartition(Hpositive, -5)[-5:]
		Hmean = np.mean(Hpositive[Hmax])

		Vpositive = abs(df['Vacc'].to_numpy())
		Vmax = np.argpartition(Vpositive, -5)[-5:]
		Vmean = np.mean(Vpositive[Vmax])

		Hind = np.argpartition(Hf, -1)[-1:]*10
		Vind = np.argpartition(Vf, -1)[-1:]*10
		Hmeans.append(Hmean)
		Vmeans.append(Vmean)
		Hfrequencies.append(int(Hind))
		Vfrequencies.append(int(Vind))
    
	#AElabeling
	df = pd.read_csv('../../data/MergedTrain_files/Bearing1_1.csv')
	X_freq = ToFrequency(df)
	X_truncated = X_freq[:int(len(X_freq)*0.8),:]
	Yb1_1 = get_AElabels(X_truncated, X_freq)

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,8))
	fig.subplots_adjust(hspace=0.4)

	x = list(range(1, len(files)+1))
	plt.xlabel("Observation")

	ax1.plot(x, Hfrequencies, '+', markersize=3, label='Horizontal')
	ax1.plot(x, Vfrequencies, 'x', markersize=3, label='Vertical')

	cuts = [386,1464,2473]

	ax1.plot(subplots=True, figsize=(6, 6))
	ax1.set_ylabel("Frequency (Hz)")
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.axvline(cuts[0], lw=0.5, zorder=0, clip_on=False)
	ax1.axvline(cuts[1], lw=0.5, zorder=0, clip_on=False)
	ax1.axvline(cuts[2], lw=0.5, zorder=0, clip_on=False)
	ax1.text(cuts[0]*1.01, 8500, '{}'.format(cuts[0]), fontsize=7)
	ax1.text(cuts[1]*1.01, 3000, '{}'.format(cuts[1]), fontsize=7)
	ax1.text(cuts[2]*1.01, 3000, '{}'.format(cuts[2]), fontsize=7)

	ax2.plot(x,Hmeans, '-', linewidth=0.8, label='Horizontal')
	ax2.plot(x,Vmeans, '-', linewidth=0.8, label='Vertical')

	ax2.plot(subplots=True, figsize=(6, 8))
	ax2.set_ylabel("Avg. Max Acceleration")
	ax2.legend(loc='best')
	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.axvline(cuts[0], lw=0.5, zorder=0, clip_on=False)
	ax2.axvline(cuts[1], lw=0.5, zorder=0, clip_on=False)
	ax2.axvline(cuts[2], lw=0.5, zorder=0, clip_on=False)

	ax3.plot(Yb1_1, '.')
	ax3.plot(subplots=True, figsize=(6, 8))
	ax3.spines['right'].set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.set_ylabel("Degradation stage")
	ax3.axvline(cuts[0], lw=0.5, zorder=0, clip_on=False)
	ax3.axvline(cuts[1], lw=0.5, zorder=0, clip_on=False)
	ax3.axvline(cuts[2], lw=0.5, zorder=0, clip_on=False)

	plt.savefig('../../reports/figures/' + b + '.png', dpi=100)
	plt.close()