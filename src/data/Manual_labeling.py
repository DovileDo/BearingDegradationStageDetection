import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal
import pandas as pd
import numpy as np
import os

# Number of samples in normalized_tone
SAMPLE_RATE = 25600
DURATION = 0.1
N = int(SAMPLE_RATE * DURATION)

Hmeans = []
Vmeans = []
Hfrequencies = []
Vfrequencies = []

b = 'Bearing1_1'

rootdir = '../../data/raw/Train_set' + b
for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		print (file)
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
		

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.5)

x = list(range(1, len(files)+1))
plt.xlabel("Time (t x 10 seconds since the start of the experiment)")

ax1.plot(x, Hfrequencies, '+', markersize=3, label='Horizontal')
ax1.plot(x, Vfrequencies, 'x', markersize=3, label='Vertical')

cuts = [386,1464,2473]

ax1.plot(subplots=True, figsize=(6, 6))
ax1.set_ylabel("Frequency (Hz)")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.axvline(cuts[0], c='b', lw=0.5, zorder=0, clip_on=False)
ax1.axvline(cuts[1], c='b', lw=0.5, zorder=0, clip_on=False)
ax1.axvline(cuts[2], c='b', lw=0.5, zorder=0, clip_on=False)
ax1.text(cuts[0]*1.01, 8000, '{}, anomaly 1'.format(cuts[0]), fontsize=7)
ax1.text(cuts[1]*1.01, 4000, '{}, anomaly 2'.format(cuts[1]), fontsize=7)
ax1.text(cuts[2]*1.01, 5000, '{}, anomaly 3'.format(cuts[2]), fontsize=7)

ax2.plot(x,Hmeans, '-', linewidth=0.8, label='Horizontal')
ax2.plot(x,Vmeans, '-', linewidth=0.8, label='Vertical')

ax2.plot(subplots=True, figsize=(6, 6))
ax2.set_ylabel("Avg. Max Acceleration")
ax2.legend(loc='best')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.axvline(cuts[0], c='b', lw=0.5, zorder=0, clip_on=False)
ax2.axvline(cuts[1], c='b', lw=0.5, zorder=0, clip_on=False)
ax2.axvline(cuts[2], c='b', lw=0.5, zorder=0, clip_on=False)

#plt.show()
plt.savefig('../../reports/labels/' + b + '.png')