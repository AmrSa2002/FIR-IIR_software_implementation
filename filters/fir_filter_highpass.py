import numpy as np
from scipy import signal

def design_highpass_fir_filter(cutoff_freq, numtaps):
    """
    Designs a high-pass FIR filter using predefined parameters.

    Parameters:
        cutoff_freq (float): Normalized cutoff frequency (0 to 1, where 1 corresponds to the Nyquist frequency).
        numtaps (int): Number of coefficients (length of the filter).

    Returns:
        numpy.ndarray: Array of FIR filter coefficients.
    """    
    # Design a high-pass FIR filter using the Hamming window method
    fir_coeff = signal.firwin(numtaps, cutoff=cutoff_freq, pass_zero=False)   # pass_zero=False specifies high-pass
    return fir_coeff


def manual_design_highpass_fir_filter(cutoff_freq, numtaps):
    """
    Manually designs a high-pass FIR filter using a specified cutoff frequency and number of taps.

    Parameters:
        cutoff_freq (float): Normalized cutoff frequency (0 to 1, where 1 corresponds to the Nyquist frequency).
        numtaps (int): Number of coefficients (length of the filter).

    Returns:
        numpy.ndarray: Array of manually computed FIR filter coefficients.
    """
    wc = 1 * np.pi * cutoff_freq # Normalized cutoff in radians

    # Calculate the middle index (center of the symmetric filter)
    M = numtaps // 2

    # Initialize the impulse response array
    h = np.zeros(numtaps)

    # Calculate the ideal impulse response for a high-pass filter
    for n in range(numtaps):
        if n == M:  # Handle the middle index
            h[n] = 1 - wc / np.pi
        else:
            h[n] = -np.sin(wc * (n - M)) / (np.pi * (n - M))
        # Apply a window function (e.g., Hamming window)
        h[n] *= 0.54 - 0.46 * np.cos(2 * np.pi * n / (numtaps - 1))
    return h
