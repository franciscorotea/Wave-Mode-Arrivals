import numpy as np
import matplotlib.pyplot as plt
import pywt
import modearrivals

# Load an Acoustic Emission sample test signal.

waveform = np.loadtxt('example_data.txt')

# Calculate scale values from frequencies of interest.

# fc: Center frequency of Morlet wavelet.
# fs: Sampling frequency (5 MHz).
# fa: Frequency value corresponding to the scale value `scale`.

fc = 5/(2*np.pi)
fs = 5000000
fa = np.array([40000, 60000, 80000, 100000, 120000, 150000, 180000, 220000, 
               270000, 310000, 360000, 420000, 480000])

scale = fc*fs/fa

# Perform continuous wavelet transform on data.

coef, freqs = pywt.cwt(waveform, scale, 'morl', 1/fs);

# Compute flexural and extensional mode arrival from decomposition 
# signals.

# For extensional mode, use only the 9th component (310 kHz). For
# flexural mode, combine several wavelet components (80 to 180 kHz).

signal_ext = coef[9].T
signal_flex = coef[2].T*coef[3].T*coef[4].T*coef[5].T*coef[6].T

flex_arrival_idx = modearrivals.get_flexure_arrival(signal_flex)
ext_arrival_idx = modearrivals.get_extension_arrival(signal_ext)

flex_arrival_time = flex_arrival_idx/fs*1e6     # in µs
ext_arrival_time = ext_arrival_idx/fs*1e6      # in µs

# Print results.

print(f'Flexural Arrival Index = {flex_arrival_idx}')
print(f'Flexural Arrival Time = {np.around(flex_arrival_time, 1)} µs\n')

print(f'Extensional Arrival Index = {ext_arrival_idx}')
print(f'Extensional Arrival Time = {np.around(ext_arrival_time, 1)} µs')

# Plot results.

# Generate a time vector for plotting.

time_vector = np.linspace(0, len(waveform)/fs*1e6, len(waveform))

# Plot AE time signal with extensional and flexural modes arrival.

fig1, ax1 = plt.subplots()

ax1.plot(time_vector, 
         waveform, 
         color='black', 
         label='AE signal')

ax1.axvline(x=flex_arrival_time, 
            linewidth=1.5, 
            linestyle='--', 
            color='#377eb8', 
            label='Flexural mode arrival = '
                  f'{np.around(flex_arrival_time, 1)} µs')

ax1.axvline(x=ext_arrival_time, 
            linewidth=1.5, 
            linestyle=':', 
            color='#e41a1c', 
            label='Extensional mode arrival = '
                  f'{np.around(ext_arrival_time, 1)} µs')

ax1.set_title('Arrival of extensional and flexural wave modes')
           
ax1.set_xlabel('Time [us]')
ax1.set_ylabel('Signal amplitude [V]')

ax1.legend()
ax1.grid()

fig1.tight_layout()

# Plot extensional decomposition signal.

fig2, ax2 = plt.subplots()

ax2.plot(time_vector, 
         signal_ext, 
         color='black', 
         label='Extensional decomposition')

ax2.axvline(x=ext_arrival_time, 
            linewidth=1.5, 
            linestyle=':', 
            color='#e41a1c', 
            label='Extensional mode arrival = '
                  f'{np.around(ext_arrival_time, 1)} µs')

ax2.set_title('Extensional decomposition signal (310 kHz component)')
           
ax2.set_xlabel('Time [µs]')
ax2.set_ylabel('Wavelet coefficient')

ax2.legend()
ax2.grid()

fig2.tight_layout()

# Plot flexural decomposition signal.

fig3, ax3 = plt.subplots()

ax3.plot(time_vector, 
         signal_flex, 
         color='black', 
         label='Flexural decomposition')

ax3.axvline(x = flex_arrival_time, 
            linewidth=1.5, 
            linestyle='--', 
            color='#377eb8', 
            label='Flexural mode arrival = '
                  f'{np.around(flex_arrival_time, 1)} µs')

ax3.set_title('Flexural decomposition signal (several wavelet components)')
           
ax3.set_xlabel('Time [µs]')
ax3.set_ylabel('Wavelet coefficient')

ax3.legend()
ax3.grid()

fig3.tight_layout()

plt.show()
