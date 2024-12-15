import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz

class FilterErrorBp(Exception):
    """Custom exception for filter errors."""
    pass

def validate_inputs(order, cutoff, fs):
    if not isinstance(order, int) or order <= 0:
        raise FilterErrorBp("Order must be a positive integer.")
    if not isinstance(cutoff, (int, float)) or cutoff <= 0:
        raise FilterErrorBp("Cutoff frequency must be a positive number.")
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise FilterErrorBp("Sampling frequency must be a positive number.")
    if cutoff >= fs / 2:
        raise FilterErrorBp("Cutoff frequency must be less than half the sampling frequency.")

def butterworth_bp_manual(lowcut: float, highcut: float, fs: float, order=4) -> tuple:
    """
    Generates Butterworth bandpass filter coefficients manually with proper normalization.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter.
    highcut (float): The upper cutoff frequency of the filter.
    fs (float): The sampling frequency.
    order (int): The order of the filter.

    Returns:
    tuple: Numerator (b) and denominator (a) coefficients of the bandpass filter.

    Example:
    >>> lowcut = 50.0  # Lower cutoff frequency (Hz)
    >>> highcut = 150.0  # Upper cutoff frequency (Hz)
    >>> fs = 500.0  # Sampling frequency (Hz)
    >>> order = 4  # Filter order
    >>> b, a = butterworth_bp_manual(lowcut, highcut, fs, order)
    >>> print(b)
    >>> print(a)

    """
    nyquist = 0.5 * fs

    # Normalizing the low and high cutoff frequencies to the Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Calculating the analog cutoff frequencies in rad/s
    wc_low = 2 * np.pi * low
    wc_high = 2 * np.pi * high

    # Calculating the center frequency (geometric mean of low and high cutoff frequencies)
    wc_center = np.sqrt(wc_low * wc_high)

    # Bandwidth of the filter
    bw = wc_high - wc_low

    # Compute analog poles for Butterworth filter
    poles = []
    for k in range(1, order + 1):
        # Calculate the angle for each pole based on the formula
        angle = np.pi * (2 * k - 1) / (2 * order)
        # Calculate the pole in the analog domain
        pole = wc_center * np.exp(1j * angle)
        poles.append(pole)

    # Convert poles to the digital domain using the bilinear transformation
    z_poles = [(2 * fs + p) / (2 * fs - p) for p in poles]
    a = np.poly(z_poles).real

    # Initialize the numerator coefficients with zeros
    b = np.zeros_like(a)
    # Set the first numerator coefficient to bw^order
    b[0] = bw ** order

    return b, a

def butterworth_bp_builtin(lowcut: float, highcut: float, fs: float, order=4) -> tuple:
    """
    Designs an IIR bandpass Butterworth filter.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter.
    highcut (float): The higher cutoff frequency of the filter.
    fs (float): The sampling frequency.
    order (int, optional): The order of the filter (default is 4).

    Returns:
    tuple: Numerator (b) and denominator (a) polynomials of the IIR filter.

    Example:
    >>> lowcut = 500.0  # Lower cutoff frequency in Hz
    >>> highcut = 1500.0  # Higher cutoff frequency in Hz
    >>> fs = 10000.0  # 10 kHz sampling frequency
    >>> b, a = butterworth_bp_builtin(lowcut, highcut, fs)
    >>> print(b)
    >>> print(a)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a



def plot_frequency_response(b1, a1, b2, a2, fs, lowcut, highcut) -> None:
    """
    Plots the frequency response of manually generated and built-in Butterworth band-pass filters.
    """

    w, h = freqz(b, a, worN=8000)
    freqs = (w / (2 * np.pi)) * fs
    magnitude = 20 * np.log10(abs(h))

    w1, h1 = np.linspace(0, np.pi, 8000, retstep=True)
    w1 = np.linspace(0, np.pi, 8000)
    h1 = np.polyval(b, np.exp(-1j * w1)) / np.polyval(a, np.exp(-1j * w1))
    freq1 = w1 * fs / (2 * np.pi)
    magnitude1 = 20 * np.log10(np.abs(h1))


    plt.figure(figsize=(8, 5))
    plt.plot(freqs, magnitude, label="Built-in", linestyle='-')
    plt.plot(freq1, magnitude1, label="Manual", linestyle='--')


    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid()
    plt.legend()
    plt.show()

# Filter parameters
fs = 500.0  # Sampling frequency
lowcut = 50.0  # The lower cutoff frequency of the filter.
highcut = 150.0  # The upper cutoff frequency of the filter.
order = 4  # Filter order

# Designing filter using scipy butter
b1, a1 = butterworth_bp_builtin(lowcut, highcut, fs, order)

# Designing filter manually
b2, a2 = butterworth_bp_manual(lowcut, highcut, fs, order)

# Plotting frequency response
plot_frequency_response(b1, a1, b2, a2, fs, lowcut, highcut)
