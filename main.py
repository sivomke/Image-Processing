import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math


samples = np.loadtxt('f20.txt', delimiter=' ')  # samples
N = samples.size  # number of data points
dt = 0.01  # sampling rate
T = 5  # [0, T] duration
t = np.array([0 + i*dt for i in range(N)])

plt.plot(t, samples)
plt.title("Input signal")
plt.show()
fft = np.fft.fft(samples)
df = 1./T  # base frequency
half = int(0.5*N) + 1
f_arr = np.array([i*df for i in range(half)])
abs_fft = abs(fft)[:half]
plt.plot(f_arr, abs_fft)
plt.title("Discrete Fourier Transform")
plt.xlabel("frequency, Hz")
plt.show()


# extracting frequencies corresponding to peaks that exceed threshold in their neighbourhood

peaks = scipy.signal.find_peaks(abs_fft)[0]
freq = f_arr[peaks][1:]  # local peaks, we exclude zero frequency
k = freq.size
print("Frequencies: {}".format(freq))


# we approximate input signal in the following class of functions
#  y(t) = a0 + a1*t + a2*t^2 + a3*t^3 + sum from i = 4 to (4 + k - 1) sin (2*pi*f[i-4]*t)
# a_k array

a_size = 4 + k

tmp_arr = np.ones((N, 1))
tmp_arr = np.append(tmp_arr, [[x] for x in t], axis = 1)
tmp_arr = np.append(tmp_arr, [[x**2] for x in t], axis = 1)
tmp_arr = np.append(tmp_arr, [[x**3] for x in t], axis = 1)

for i in range(k):
    tmp_arr = np.append(tmp_arr, [[math.sin(2*np.pi*freq[i]*x)] for x in t], axis = 1)


a_coeffs = np.linalg.lstsq(tmp_arr, samples, rcond=None)[0]

print(a_coeffs)

def y(t, coeffs):
    trig = np.array([math.sin(2*math.pi*f*t) for f in freq])
    trig_coeffs = np.array(coeffs[4:])
    dummy = np.dot(trig, trig_coeffs)
    return coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + dummy


approx_samples = [y(x, a_coeffs) for x in t]
plt.plot(t, approx_samples,  label='signal approximation')
plt.plot(t, samples, label='input signal')

plt.title("Approximated signal")
plt.show()


