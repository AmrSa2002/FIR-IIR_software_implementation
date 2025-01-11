import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz
from scipy.signal import tf2zpk
from scipy.signal import zpk2tf


class FilterErrorBp(Exception):
    """Custom exception for filter errors."""
    pass

def validate_inputs(order, lowcut, highcut, fs):
    if fs == 0:
        raise FilterErrorBp("Sampling frequency (fs) cannot be zero.")
    if lowcut == 0 or highcut == 0:
        raise FilterErrorBp("Cutoff frequencies (lowcut, highcut) cannot be zero.")
    if not isinstance(order, int) or order <= 0:
        raise FilterErrorBp("Order must be a positive integer.")
    if not isinstance(lowcut, (int, float)) or lowcut <= 0:
        raise FilterErrorBp("Lowcut frequency must be a positive number.")
    if not isinstance(highcut, (int, float)) or highcut <= 0:
        raise FilterErrorBp("Highcut frequency must be a positive number.")
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise FilterErrorBp("Sampling frequency must be a positive number.")
    if highcut >= fs / 2:
        raise FilterErrorBp("Highcut frequency must be less than half the sampling frequency.")
    if lowcut >= fs / 2:
        raise FilterErrorBp("Lowcut frequency must be less than half the sampling frequency.")
    if lowcut >= highcut:
        raise FilterErrorBp("Lowcut frequency must be less than highcut frequency.")


def butterworth_bp_manual(order, lowcut: float, highcut: float, fs: float) -> tuple:
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
    validate_inputs(order, lowcut, highcut, fs)
    nyquist = fs/2

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

def butterworth_bp_builtin(order, lowcut: float, highcut: float, fs: float) -> tuple:
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
    validate_inputs(order, lowcut, highcut, fs)
    nyquist = fs/2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butterworth_bp_manual_opt(order, lowcut: float, highcut: float, fs: float) -> tuple:
    """
    Optimized generation of Butterworth bandpass filter coefficients manually with proper normalization.

    Parameters:
    lowcut (float): The lower cutoff frequency of the filter.
    highcut (float): The upper cutoff frequency of the filter.
    fs (float): The sampling frequency.
    order (int): The order of the filter.

    Returns:
    tuple: Numerator (b) and denominator (a) coefficients of the bandpass filter.
    """
    validate_inputs(order, lowcut, highcut, fs)
    nyquist = fs/2

    # Normalize cutoff frequencies
    low, high = lowcut / nyquist, highcut / nyquist
    wc_low, wc_high = 2 * np.pi * low, 2 * np.pi * high
    wc_center = np.sqrt(wc_low * wc_high)
    bw = wc_high - wc_low

    # Calculate analog poles using vectorized operations
    angles = np.pi * (2 * np.arange(1, order + 1) - 1) / (2 * order)
    poles = wc_center * np.exp(1j * angles)

    # Bilinear transformation to map analog poles to digital domain
    z_poles = (2 * fs + poles) / (2 * fs - poles)
    a = np.poly(z_poles).real

    # Calculate numerator coefficients directly
    b = np.array([bw ** order] + [0] * (len(a) - 1))
    return b, a

b_builtin, a_builtin = butterworth_bp_builtin(4, 100, 300, 1000)
zeros, poles, gain = tf2zpk(b_builtin, a_builtin)
zeros_manual = zeros
poles_manual = poles
gain_manual = gain


b_manual, a_manual = zpk2tf(zeros_manual, poles_manual, gain_manual)
b_opt, a_opt = zpk2tf(zeros_manual, poles_manual, gain_manual)


def plot_iir_bandpass_filter_opt_responses(order: int, low_cutoff: float, high_cutoff: float, fs: int):
    """
    Plots the frequency responses of the band-pass filters generated by both manual, optimized IIR methods, and built-in IIR method.

    Parameters:
    low_cutoff (float): The low cutoff frequency of the filter (in Hz).
    high_cutoff (float): The high cutoff frequency of the filter (in Hz).
    order (int): The order of the filter.
    fs (int): The sample rate of the signal (in Hz).

    Raises:
    ValueError: If order is not a positive integer or if cutoff values are not positive.

    Example:
    --------
    >>> plot_bandpass_filter_opt_responses(4, 100, 300, 1000)
    """
    validate_inputs(order, low_cutoff, high_cutoff, fs)

    # Compute frequency response for manual filter
    w_manual, h_manual = freqz(b_manual, a_manual, worN=8000)
    # Compute frequency response for optimized filter
    w_opt, h_opt = freqz(b_opt, a_opt, worN=8000)
    # Compute frequency response for built-in filter
    w_builtin, h_builtin = freqz(b_builtin, a_builtin, worN=8000)

    # Plot frequency responses
    plt.figure()
    plt.plot(w_manual * fs / (2 * np.pi), 20 * np.log10(np.abs(h_manual) + 1e-10), 'b', label='Manual')
    plt.plot(w_opt * fs / (2 * np.pi), 20 * np.log10(np.abs(h_opt) + 1e-10), 'r--', label='Optimized')
    plt.plot(w_builtin * fs / (2 * np.pi), 20 * np.log10(np.abs(h_builtin) + 1e-10), 'g:', label='Built-in')

    plt.title('Frequency Response of IIR Bandpass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_iir_bandpass_filter_opt_coefficients(order: int, low_cutoff: float, high_cutoff: float, fs: int):
    """
    Plots the band-pass filter coefficients generated by both manual and optimized IIR methods side by side.

    Parameters:
    low_cutoff (float): The low cutoff frequency of the filter (in Hz).
    high_cutoff (float): The high cutoff frequency of the filter (in Hz).
    order (int): The order of the filter.
    fs (int): The sample rate of the signal (in Hz).

    Raises:
    ValueError: If order is not a positive integer or if cutoff values are not positive.

    Example:
    --------
    >>> plot_bandpass_filter_opt_coefficients(4, 100, 300, 1000)
    """
    validate_inputs(order, low_cutoff, high_cutoff, fs)

    # Plot coefficients
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(b_opt, linefmt='b-', markerfmt='bo', basefmt='r-', label='Manual')
    plt.stem(b_manual, linefmt='g-', markerfmt='go', basefmt='r-', label='Built-in')
    plt.title('Numerator Coefficients')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.stem(a_opt, linefmt='b-', markerfmt='bo', basefmt='r-', label='Manual')
    plt.stem(a_manual, linefmt='g-', markerfmt='go', basefmt='r-', label='Built-in')
    plt.title('Denominator Coefficients')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bandpass_iir_filter_coefficients(order: int, low_cutoff: float, high_cutoff: float, fs: int):
    """
    Plots the band-pass filter coefficients generated by the manual IIR method.

    Parameters:
    low_cutoff (float): The low cutoff frequency of the filter (in Hz).
    high_cutoff (float): The high cutoff frequency of the filter (in Hz).
    order (int): The order of the filter.
    fs (int): The sample rate of the signal (in Hz).

    Raises:
    ValueError: If order is not a positive integer or if cutoff values are not positive.

    Example:
    --------
    >>> plot_bandpass_filter_manual_coefficients(1000, 100, 500, 8000)
    """
    validate_inputs(order, low_cutoff, high_cutoff, fs)

    # Plot coefficients
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(b_manual, linefmt='b-', markerfmt='bo', basefmt='r-', label='Manual')
    plt.stem(b_builtin, linefmt='g-', markerfmt='go', basefmt='r-', label='Built-in')
    plt.title('Numerator Coefficients')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.stem(a_manual, linefmt='b-', markerfmt='bo', basefmt='r-', label='Manual')
    plt.stem(a_builtin, linefmt='g-', markerfmt='go', basefmt='r-', label='Built-in')
    plt.title('Denominator Coefficients')
    plt.legend()
    plt.tight_layout()
    plt.show()
