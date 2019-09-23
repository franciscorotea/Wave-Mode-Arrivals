import numpy as np
import matplotlib.pyplot as plt
import pywt
import waveModeArrivals as wm

# Load an Acoustic Emission sample test signal.

waveform = np.loadtxt('exampleData.txt')

# Calculate scale values from frequencies of interest.

fc = 5/(2*np.pi)
fs = 5000000
fa = np.array([40000, 60000, 80000, 100000, 120000, 150000, 180000, 220000, 270000, 310000, 360000, 420000, 480000])

scale = fc*fs/fa

# Perform continuous wavelet transform on the data.

coef, freqs = pywt.cwt(waveform, scale, 'morl', 1/fs);

# Compute flexural and extensional mode arrival from decomposition signals.

signalExt = coef[9].T                                               # For extensional mode, we use only the 9th component (310 kHz)
signalFlex = coef[2].T*coef[3].T*coef[4].T*coef[5].T*coef[6].T      # For flexural mode, we combine several wavelet components (80 to 180 kHz)

flexArrivalTime = wm.getFlexureArrival(signalFlex)/fs*1e6
extArrivalTime = wm.getExtensionArrival(signalExt)/fs*1e6

print('Flexural Arrival Index =', int(flexArrivalTime*fs/1e6))
print('Flexural Arrival Time =', np.around(flexArrivalTime, 1), 'us')
print()
print('Extensional Arrival Index =', int(extArrivalTime*fs/1e6))
print('Extensional Arrival Time =', np.around(extArrivalTime, 1), 'us')

# Generate a time vector for plotting.

timeVector = np.linspace(0, len(waveform)/fs*1e6, len(waveform))

# Plot AE time signal with extensional and flexural modes arrival.

fig1, ax1 = plt.subplots()

ax1.plot(timeVector, waveform, color = 'black', linewidth = 1, label = 'AE signal')
ax1.axvline(x = flexArrivalTime, linewidth = 1.5, linestyle = '--', color = '#377eb8', label = 'Flexural mode = %.1f us' % flexArrivalTime)
ax1.axvline(x = extArrivalTime, linewidth = 1.5, linestyle = ':', color = '#e41a1c', label = 'Extensional mode = %.1f us' % extArrivalTime)

ax1.set_title('Arrival of extensional and flexural wave modes')
           
ax1.set_xlabel('Time [us]')
ax1.set_ylabel('Signal amplitude [V]')

ax1.legend()
ax1.grid()

fig1.tight_layout()

# Plot extensional decomposition signal.

fig2, ax2 = plt.subplots()

ax2.plot(timeVector, signalExt, color = 'black', linewidth = 1, label = 'Extensional decomposition')
ax2.axvline(x = flexArrivalTime, linewidth = 1.5, linestyle = '--', color = '#377eb8', label = 'Flexural mode = %.1f us' % flexArrivalTime)
ax2.axvline(x = extArrivalTime, linewidth = 1.5, linestyle = ':', color = '#e41a1c', label = 'Extensional mode = %.1f us' % extArrivalTime)

ax2.set_title('Extensional decomposition signal (310 kHz component)')
           
ax2.set_xlabel('Time [us]')
ax2.set_ylabel('Wavelet coefficient')

ax2.legend()
ax2.grid()

fig2.tight_layout()

# Plot flexural decomposition signal.

fig3, ax3 = plt.subplots()

ax3.plot(timeVector, signalFlex, color = 'black', linewidth = 1, label = 'Flexural decomposition')
ax3.axvline(x = flexArrivalTime, linewidth = 1.5, linestyle = '--', color = '#377eb8', label = 'Flexural mode = %.1f us' % flexArrivalTime)
ax3.axvline(x = extArrivalTime, linewidth = 1.5, linestyle = ':', color = '#e41a1c', label = 'Extensional mode = %.1f us' % extArrivalTime)

ax3.set_title('Flexural decomposition signal (several wavelet components)')
           
ax3.set_xlabel('Time [us]')
ax3.set_ylabel('Wavelet coefficient')

ax3.legend()
ax3.grid()

fig3.tight_layout()

plt.show()