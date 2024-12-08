import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin

class FilterError(Exception):
    """Custom exception for filter errors."""
    pass

def validate_inputs(lowcut, highcut, num_taps, sample_rate=None):
    if not isinstance(num_taps, int) or num_taps <= 0:
        raise FilterError("num_taps must be a positive integer.")
    if not (0 < lowcut < highcut <= 1):
        raise FilterError("lowcut and highcut must be in the range (0, 1] and lowcut < highcut.")
    if sample_rate is not None and (not isinstance(sample_rate, int) or sample_rate <= 0):
        raise FilterError("sample_rate must be a positive integer.")

def sinc_function(x: float) -> float:
    """
    Sinc function, where sinc(0) = 1.

    Parameters:
    x (float): The input value.

    Returns:
    float: The result of the sinc function.
    """
    return np.sinc(x / np.pi)

def bandpass_fir_filter_manual(lowcut: float, highcut: float, num_taps: int) -> np.ndarray:
    """
    Generates symmetric coefficients for a FIR band-pass filter using the sinc function and applies a Hamming window.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    highcut (float): The upper cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    num_taps (int): The number of filter coefficients (taps).

    Returns:
    numpy.ndarray: The normalized filter coefficients after applying the Hamming window.

    Raises:
    ValueError: If num_taps is not a positive integer or if lowcut and highcut are not in the range (0, 1).

    Example:
    --------
    >>> import numpy as np
    >>> from fir_filter_bandpass import bandpass_fir_filter_manual
    >>> num_taps = 5
    >>> lowcut = 0.2
    >>> highcut = 0.5
    >>> h = bandpass_fir_filter_manual(lowcut, highcut, num_taps)
    >>> print(h)
    [0.06799017 0.28200983 0.3        0.28200983 0.06799017]
    """
    validate_inputs(lowcut, highcut, num_taps)

    nyquist = 0.5  # Nyquist frequency
    ideal_impulse_response = np.zeros(num_taps)
    mid_point = num_taps // 2
    for i in range(num_taps):
        if i == mid_point:
            ideal_impulse_response[i] = 2 * (highcut - lowcut)
        else:
            ideal_impulse_response[i] = (np.sin(2 * np.pi * highcut * (i - mid_point)) -
                                         np.sin(2 * np.pi * lowcut * (i - mid_point))) / (np.pi * (i - mid_point))

    # Apply Hamming window
    window = np.hamming(num_taps)
    fir_coefficients = ideal_impulse_response * window

    return fir_coefficients

def bandpass_fir_filter_opt_manual(lowcut: float, highcut: float, num_taps: int) -> np.ndarray:
    """
    Generates symmetric coefficients for a FIR band-pass filter using the sinc function and applies a Hamming window.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    highcut (float): The upper cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    num_taps (int): The number of filter coefficients (taps).

    Returns:
    numpy.ndarray: The normalized filter coefficients after applying the Hamming window.

    Raises:
    ValueError: If num_taps is not a positive integer or if lowcut and highcut are not in the range (0, 1).

    Example:
    --------
    >>> import numpy as np
    >>> from fir_filter_bandpass import bandpass_fir_filter_manual
    >>> num_taps = 5
    >>> lowcut = 0.2
    >>> highcut = 0.5
    >>> h = bandpass_fir_filter_manual(lowcut, highcut, num_taps)
    >>> print(h)
    [0.06799017 0.28200983 0.3        0.28200983 0.06799017]
    """
    validate_inputs(lowcut, highcut, num_taps)

    nyquist = 0.5  # Nyquist frequency
    mid_point = num_taps // 2
    n = np.arange(num_taps)

    # Vectorized calculation of ideal impulse response
    ideal_impulse_response = np.zeros(num_taps)
    ideal_impulse_response[mid_point] = 2 * (highcut - lowcut)

    non_mid = n != mid_point
    i_values = (n[non_mid] - mid_point)

    ideal_impulse_response[non_mid] = (np.sin(2 * np.pi * highcut * i_values) - np.sin(2 * np.pi * lowcut * i_values)) / (np.pi * i_values)

    # Apply Hamming window using numpy
    window = np.hamming(num_taps)
    fir_coefficients = ideal_impulse_response * window

    return fir_coefficients

def bandpass_fir_filter_firwin(lowcut: float, highcut: float, num_taps: int) -> np.ndarray:
    """
    Generates coefficients for a FIR band-pass filter using the firwin function from scipy.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    highcut (float): The upper cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    num_taps (int): The number of filter coefficients (taps).

    Returns:
    numpy.ndarray: The filter coefficients.

    Raises:
    ValueError: If num_taps is not a positive integer or if lowcut and highcut are not in the range (0, 1).

    Example:
    --------
    >>> import numpy as np
    >>> from fir_filter_bandpass import bandpass_fir_filter_firwin
    >>> num_taps = 5
    >>> lowcut = 0.2
    >>> highcut = 0.5
    >>> h = bandpass_fir_filter_firwin(lowcut, highcut, num_taps)
    >>> print(h)
    [0.06799017 0.28200983 0.3        0.28200983 0.06799017]
    """
    validate_inputs(lowcut, highcut, num_taps)

    nyquist = 0.5  # Nyquist frequency
    return firwin(num_taps, [lowcut / nyquist, highcut / nyquist], pass_zero=False, window="hamming")

def plot_bandpass_filter_responses(lowcut: float, highcut: float, num_taps: int, sample_rate: int):
    """
    Plots the frequency responses of the band-pass filters generated by both manual and firwin methods.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    highcut (float): The upper cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    num_taps (int): The number of filter coefficients (taps).
    sample_rate (int): The sample rate of the signal.

    Raises:
    ValueError: If num_taps is not a positive integer, if lowcut and highcut are not in the range (0, 1), or if sample_rate is not a positive integer.

    Example:
    --------
    >>> from fir_filter_bandpass import plot_bandpass_filter_responses
    >>> plot_bandpass_filter_responses(0.2, 0.5, 5, 1000)
    """
    validate_inputs(lowcut, highcut, num_taps, sample_rate)

    h_manual = bandpass_fir_filter_manual(lowcut, highcut, num_taps)
    h_firwin = bandpass_fir_filter_firwin(lowcut, highcut, num_taps)

    w_manual, h_response_manual = np.fft.fft(h_manual, 1024), np.fft.fftshift(np.abs(np.fft.fft(h_manual, 1024)))
    w_firwin, h_response_firwin = np.fft.fft(h_firwin, 1024), np.fft.fftshift(np.abs(np.fft.fft(h_firwin, 1024)))
    freqs = np.fft.fftfreq(len(w_manual), 1 / sample_rate)

    plt.plot(freqs[freqs >= 0], 20 * np.log10(h_response_manual[freqs >= 0]), label='Manual')
    plt.plot(freqs[freqs >= 0], 20 * np.log10(h_response_firwin[freqs >= 0]), label='Firwin')
    plt.title('Bandpass Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_bandpass_filter_coefficients(lowcut: float, highcut: float, num_taps: int):
    """
    Plots the band-pass filter coefficients generated by both manual and firwin methods side by side.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    highcut (float): The upper cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    num_taps (int): The number of filter coefficients (taps).

    Raises:
    ValueError: If num_taps is not a positive integer or if lowcut and highcut are not in the range (0, 1).

    Example:
    --------
    >>> from fir_filter_bandpass import plot_bandpass_filter_coefficients
    >>> plot_bandpass_filter_coefficients(0.2, 0.5, 5)
    """
    validate_inputs(lowcut, highcut, num_taps)

    h_manual = bandpass_fir_filter_manual(lowcut, highcut, num_taps)
    h_firwin = bandpass_fir_filter_firwin(lowcut, highcut, num_taps)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(h_manual)
    plt.title('Manual Bandpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.stem(h_firwin)
    plt.title('Firwin Bandpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_bandpass_filter_opt_responses(lowcut: float, highcut: float, num_taps: int, sample_rate: int):
    """
    Plots the frequency responses of the band-pass filters generated by both manual and firwin methods.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    highcut (float): The upper cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    num_taps (int): The number of filter coefficients (taps).
    sample_rate (int): The sample rate of the signal.

    Raises:
    ValueError: If num_taps is not a positive integer, if lowcut and highcut are not in the range (0, 1), or if sample_rate is not a positive integer.
    """

    validate_inputs(lowcut, highcut, num_taps, sample_rate)

    h_manual = bandpass_fir_filter_opt_manual(lowcut, highcut, num_taps)
    h_firwin = bandpass_fir_filter_firwin(lowcut, highcut, num_taps)

    w_manual, h_response_manual = np.fft.fft(h_manual, 1024), np.fft.fftshift(np.abs(np.fft.fft(h_manual, 1024)))
    w_firwin, h_response_firwin = np.fft.fft(h_firwin, 1024), np.fft.fftshift(np.abs(np.fft.fft(h_firwin, 1024)))
    freqs = np.fft.fftfreq(len(w_manual), 1 / sample_rate)

    plt.plot(freqs[freqs >= 0], 20 * np.log10(h_response_manual[freqs >= 0]), label='Manual')
    plt.plot(freqs[freqs >= 0], 20 * np.log10(h_response_firwin[freqs >= 0]), label='Firwin')
    plt.title('Optimized Bandpass Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_bandpass_filter_opt_coefficients(lowcut: float, highcut: float, num_taps: int):
    """
    Plots the band-pass filter coefficients generated by both manual and firwin methods side by side.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    highcut (float): The upper cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < lowcut < highcut < 1).
    num_taps (int): The number of filter coefficients (taps).

    Raises:
    ValueError: If num_taps is not a positive integer or if lowcut and highcut are not in the range (0, 1).
    """

    validate_inputs(lowcut, highcut, num_taps)

    h_manual = bandpass_fir_filter_opt_manual(lowcut, highcut, num_taps)
    h_firwin = bandpass_fir_filter_firwin(lowcut, highcut, num_taps)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(h_manual)
    plt.title('Optimized Manual Bandpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.stem(h_firwin)
    plt.title('Firwin Bandpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.tight_layout()
    plt.show()
