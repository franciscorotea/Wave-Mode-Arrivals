# Identification of the arrival times of extensional and flexural wave modes of ultrasonic signals

This code provides a Python implementation of Gupta's algorithm for the identification of the arrival times of extensional and flexural wave modes using the wavelet decomposition of ultrasonic signals. It is based on Gupta's own version of the algorithm, originally written in MATLAB. A complete explanation of the algorithm can be found in *Gupta, A. and Duke Jr, J. C., Identifying the arrival of extensional and flexural wave modes using wavelet decomposition of ultrasonic signals, Ultrasonics 82, pp. 261-271 (2018).*

## Getting Started

The code is tested with Python 3.7. Next section provides the prerequisites to run the program.

### Prerequisites

The code is dependant on the following external libraries: Numpy, PyWavelets and Matplotlib. These can be installed with Python's inbuilt package management system, [pip](https://pip.pypa.io/en/stable/). See Python's tutorial on [installing packages](https://packaging.python.org/tutorials/installing-packages/#id17) for information about this issue. In short, the installation can be made as:

```
pip install numpy
pip install PyWavelets
pip install matplotlib
```

## Sample code

Run `exampleCode.py` for a sample code of the program. This code uses the test signal `exampleData.txt`, an Acoustic Emission waveform from a pencil lead break test (normalized Hsu-Nielsen source) on a 30 x 30 x 0.1 cm CFRP plate recorded at a 5 MHz sample rate. The pencil lead was broken at the center of the plate, and the signal was acquired at a 10 cm distance. Note that the signal file does not have any header; if this was the case, you should look into the [skiprows](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.loadtxt.html) parameter of the `numpy.loadtxt` function.

### Selection of scale values

The scale values for the wavelet transform should be chosen such that the entire frequency spectrum of sensitivity is covered. For this particular signal, the frequencies chosen (in kHz) are: 40, 60, 80, 100, 120, 150, 180, 220, 270, 310, 360, 420 and 480. These frequency values are converted into the corresponding scale values using the following formula:

```
scale = fc*fs/fa
```

where `fc` is the center frequency of the Morlet wavelet, `fs` is the sample frequency, and `fa` is the array of frequencies to convert.
Then, the wavelet transform is computed, obtaining 13 sets of coefficients (one for each scale specified), each set containing the same number of samples as the original time signal. Each set of coefficients can therefore be thought to represent the portion of the signal that is centered around the frequency corresponding to the specified scale value (*wavelet decomposition*).

### Computation of arrival times

The calculation of the arrival times will be different depending on the mode:

* Extensional mode: Since the extensional mode always comprises higher frequency components, it is sufficient to use any of these components to detect the arrival of the extensional mode. In this example, the 310 kHz component is used, because of its high amplitude.

```
signalExt = coef[9].T  
```

* Flexural mode: In this case, the arrival of this mode does not simply coincide with any of the wavelet components. Nevertheless, it was found that the element-wise product of all of the wavelet components identified as ‘lower’ frequencies is useful for the calculation of the flexural mode. In this example, these lower-frequency region is found between 80 and 180 kHz.

```
signalFlex = coef[2].T*coef[3].T*coef[4].T*coef[5].T*coef[6].T
```

### Difference with MATLAB's version

It was found that this Python version can lead to slighly different results when compared to the original MATLAB counterpart. This difference is mainly attributed to the wavelet transform implementation, which is somehow different in MATLAB and the PyWavelets library. On the other hand, if the same wavelet coefficient matrix is used, the results are equal in all signals tested.

## Results

The `exampleCode.py` should return the following plots:

- Sample AE signal with the calculated extensional and flexural wave modes.

![alt text](https://i.imgur.com/dJPUtlb.png)

- Signal used for the calculation of the extensional wave mode (310 kHz component)

![alt text](https://i.imgur.com/194yT7Q.png)

- Signal used for the calculation of the flexural wave mode (element wise multiplication of several ‘low-frequency’ components, between 80 to 180 kHz)

![alt text](https://i.imgur.com/q5GiHeh.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
