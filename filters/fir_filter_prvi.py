import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin

def sinc_function(x):
    # Funkcija sinc(x), gdje je sinc(0) = 1
    return np.sinc(x / np.pi)

def generate_fir_coefficients(num_taps, cutoff_freq):
    # Generisanje simetričnih koeficijenata za FIR low-pass filter koristeći sinc funkciju
    M = num_taps - 1
    h = np.zeros(num_taps)
    for n in range(num_taps):
        if n == M / 2:
            h[n] = 2 * cutoff_freq  # Poseban slučaj kada je n = M / 2
        else:
            h[n] = (2 * cutoff_freq) * sinc_function(3.15*cutoff_freq * (n - M / 2))
        # Primjena prozora (Hamming prozor)
        h[n] *= 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    # Normalizacija koeficijenata
    h /= np.sum(h)
    return h
