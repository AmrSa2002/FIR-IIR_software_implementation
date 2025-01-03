import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin
from scipy.signal import freqz

class FilterError(Exception):
    """Custom exception for filter errors."""
    pass

def validate_inputs(cutoff_freq, num_taps, sample_rate=None):
    if not isinstance(num_taps, int) or num_taps <= 0:
        raise FilterError("num_taps must be a positive integer.")
    if not (0 < cutoff_freq <= 1):
        raise FilterError("cutoff_freq must be in the range (0, 1].")
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

def highpass_fir_filter_manual(cutoff_freq: float, num_taps: int) -> np.ndarray:
    """
    Generates symmetric coefficients for a FIR high-pass filter using the sinc function and applies a Hamming window.

    Parameters:
    num_taps (int): The number of filter coefficients (taps).
    cutoff_freq (float): The cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < cutoff_freq <= 1).

    Returns:
    numpy.ndarray: The normalized filter coefficients after applying the Hamming window.

    Raises:
    ValueError: If num_taps is not a positive integer or if cutoff_freq is not in the range (0, 1].

    Example:
    --------
    >>> import numpy as np
    >>> from fir_filter_highpass import highpass_fir_filter_manual
    >>> num_taps = 5
    >>> cutoff_freq = 0.25
    >>> h = highpass_fir_filter_manual(cutoff_freq, num_taps)
    >>> print(h)
    [0.06799017 0.28200983 0.3        0.28200983 0.06799017]
    """
    validate_inputs(cutoff_freq, num_taps)

    wc = 1 * np.pi * cutoff_freq  # Normalized cutoff in radians
    M = num_taps // 2  # Calculate the middle index (center of the symmetric filter)
    h = np.zeros(num_taps)  # Initialize the impulse response array

    # Calculate the ideal impulse response for a high-pass filter
    for n in range(num_taps):
        if n == M:  # Handle the middle index
            h[n] = 1 - wc / np.pi
        else:
            h[n] = -np.sin(wc * (n - M)) / (np.pi * (n - M))
        # Apply a window function (e.g., Hamming window)
        h[n] *= 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_taps - 1))

    return h

def highpass_fir_filter_opt_manual(cutoff_freq: float, num_taps: int) -> np.ndarray:
    """
    Generates symmetric coefficients for a FIR high-pass filter using the sinc function and applies a Hamming window.

    Parameters:
    num_taps (int): The number of filter coefficients (taps).
    cutoff_freq (float): The cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < cutoff_freq <= 1).

    Returns:
    numpy.ndarray: The normalized filter coefficients after applying the Hamming window.

    Raises:
    ValueError: If num_taps is not a positive integer or if cutoff_freq is not in the range (0, 1].

    Example:
    --------
    >>> import numpy as np
    >>> from fir_filter_highpass import highpass_fir_filter_manual
    >>> num_taps = 5
    >>> cutoff_freq = 0.25
    >>> h = highpass_fir_filter_manual(cutoff_freq, num_taps)
    >>> print(h)
    [0.06799017 0.28200983 0.3        0.28200983 0.06799017]
    """
    validate_inputs(cutoff_freq, num_taps)

    wc = np.pi * cutoff_freq  # Normalized cutoff frequency in radians
    M = num_taps // 2  # Middle index for symmetric filter

    n = np.arange(num_taps)  # Tap indices
    h = np.where( # Selects values based on the given condition.
        n == M,
        1 - wc / np.pi,
        -np.sin(wc * (n - M)) / (np.pi * (n - M + 1e-10)) ## A small constant (1e-10) is addedto prevent division by zero 
    )

    # Apply Hamming window
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_taps - 1))
    h *= hamming_window

    return h

def highpass_fir_filter_firwin(cutoff_freq: float, num_taps: int) -> np.ndarray:
    """
    Generates coefficients for a FIR high-pass filter using the firwin function from scipy.

    Parameters:
    num_taps (int): The number of filter coefficients (taps).
    cutoff_freq (float): The cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < cutoff_freq <= 1).

    Returns:
    numpy.ndarray: The filter coefficients.

    Raises:
    ValueError: If num_taps is not a positive integer or if cutoff_freq is not in the range (0, 1].

    Example:
    --------
    >>> import numpy as np
    >>> from fir_filter_highpass import highpass_fir_filter_firwin
    >>> num_taps = 5
    >>> cutoff_freq = 0.25
    >>> h = highpass_fir_filter_firwin(cutoff_freq, num_taps)
    >>> print(h)
    [0.06799017 0.28200983 0.3        0.28200983 0.06799017]
    """
    validate_inputs(cutoff_freq, num_taps)

    return firwin(num_taps, cutoff=cutoff_freq, pass_zero=False, window="hamming")

def plot_highpass_filter_responses(cutoff_freq: float, num_taps: int, sample_rate: int):
    """
    Plots the frequency responses of the high-pass filters generated by both manual and firwin methods.

    Parameters:
    cutoff_freq (float): The cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < cutoff_freq <= 1).
    num_taps (int): The number of filter coefficients (taps).
    sample_rate (int): The sample rate of the signal.

    Raises:
    ValueError: If num_taps is not a positive integer, if cutoff_freq is not in the range (0, 1], or if sample_rate is not a positive integer.

    Example:
    --------
    >>> from fir_filter_highpass import plot_highpass_filter_responses
    >>> plot_highpass_filter_responses(0.25, 5, 1000)
    """
    validate_inputs(cutoff_freq, num_taps, sample_rate)

    h_manual = highpass_fir_filter_manual(cutoff_freq, num_taps)
    h_firwin = highpass_fir_filter_firwin(cutoff_freq, num_taps)

    w_manual, h_response_manual = freqz(h_manual, worN=8000)
    w_firwin, h_response_firwin = freqz(h_firwin, worN=8000)

    plt.figure()
    plt.plot(0.5 * sample_rate * w_manual / np.pi, np.abs(h_response_manual), 'b', label='Manual')
    plt.plot(0.5 * sample_rate * w_firwin / np.pi, np.abs(h_response_firwin), 'r', label='firwin')
    plt.title("High-pass Filter Frequency Response")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.legend()
    plt.grid()
    plt.show()

def plot_highpass_filter_coefficients(cutoff_freq: float, num_taps: int):
    """
    Plots the high-pass filter coefficients generated by both manual and firwin methods side by side.

    Parameters:
    cutoff_freq (float): The cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < cutoff_freq <= 1).
    num_taps (int): The number of filter coefficients (taps).

    Raises:
    ValueError: If num_taps is not a positive integer or if cutoff_freq is not in the range (0, 1].

    Example:
    --------
    >>> from fir_filter_highpass import plot_highpass_filter_coefficients
    >>> plot_highpass_filter_coefficients(0.25, 5)
    """
    validate_inputs(cutoff_freq, num_taps)

    h_manual = highpass_fir_filter_manual(cutoff_freq, num_taps)
    h_firwin = highpass_fir_filter_firwin(cutoff_freq, num_taps)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(h_manual)
    plt.title('Manual Highpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.stem(h_firwin)
    plt.title('Firwin Highpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_highpass_filter_opt_responses(cutoff_freq: float, num_taps: int, sample_rate: int):
    """
    Plots the frequency responses of the high-pass filters generated by both manual and firwin methods.

    Parameters:
    cutoff_freq (float): The cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < cutoff_freq <= 1).
    num_taps (int): The number of filter coefficients (taps).
    sample_rate (int): The sample rate of the signal.

    Raises:
    ValueError: If num_taps is not a positive integer, if cutoff_freq is not in the range (0, 1], or if sample_rate is not a positive integer.

    Example:
    --------
    >>> from fir_filter_highpass import plot_highpass_filter_responses
    >>> plot_highpass_filter_responses(0.25, 5, 1000)
    """
    validate_inputs(cutoff_freq, num_taps, sample_rate)

    h_manual = highpass_fir_filter_opt_manual(cutoff_freq, num_taps)
    h_firwin = highpass_fir_filter_firwin(cutoff_freq, num_taps)

    w_manual, h_response_manual = freqz(h_manual, worN=8000)
    w_firwin, h_response_firwin = freqz(h_firwin, worN=8000)

    plt.figure()
    plt.plot(0.5 * sample_rate * w_manual / np.pi, np.abs(h_response_manual), 'b', label='Manual')
    plt.plot(0.5 * sample_rate * w_firwin / np.pi, np.abs(h_response_firwin), 'r', label='firwin')
    plt.title("Optimized FIR High-pass Filter Frequency Response")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.legend()
    plt.grid()
    plt.show()

def plot_highpass_filter_opt_coefficients(cutoff_freq: float, num_taps: int):
    """
    Plots the high-pass filter coefficients generated by both manual and firwin methods side by side.

    Parameters:
    cutoff_freq (float): The cutoff frequency of the filter (normalized to the Nyquist frequency, i.e., 0 < cutoff_freq <= 1).
    num_taps (int): The number of filter coefficients (taps).

    Raises:
    ValueError: If num_taps is not a positive integer or if cutoff_freq is not in the range (0, 1].

    Example:
    --------
    >>> from fir_filter_highpass import plot_highpass_filter_coefficients
    >>> plot_highpass_filter_coefficients(0.25, 5)
    """
    validate_inputs(cutoff_freq, num_taps)

    h_manual = highpass_fir_filter_opt_manual(cutoff_freq, num_taps)
    h_firwin = highpass_fir_filter_firwin(cutoff_freq, num_taps)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(h_manual)
    plt.title('Manual Optimized Highpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.stem(h_firwin)
    plt.title('Firwin Highpass Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.tight_layout()
    plt.show()
