import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz


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
    validate_inputs(order, lowcut, highcut, fs)
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
    validate_inputs(order, lowcut, highcut, fs)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

    

def butterworth_bp_manual_opt(lowcut: float, highcut: float, fs: float, order=4) -> tuple:
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
    nyquist = 0.5 * fs

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
    b = [bw ** order] + [0] * (len(a) - 1)

    return b, a




def plot_frequency_response(b, a, fs, lowcut, highcut) -> None:
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


    plt.title("Frequency Response IIR Bandpass filter")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid()
    plt.legend()
    plt.show()


def plot_iir_bandpass_filter_opt_responses(order: int, low_cutoff: float, high_cutoff: float, fs: int):
    """
    Plots the frequency responses of the band-pass filters generated by both manual and optimized IIR methods.

    Parameters:
    low_cutoff (float): The low cutoff frequency of the filter (in Hz).
    high_cutoff (float): The high cutoff frequency of the filter (in Hz).
    order (int): The order of the filter.
    fs (int): The sample rate of the signal (in Hz).
    
    Raises:
    ValueError: If order is not a positive integer or if cutoff values are not positive.
    
    Example:
    --------
    >>> plot_bandpass_filter_opt_responses(1000, 100, 500, 8000)
    """
    validate_inputs(order, low_cutoff, high_cutoff, fs)

    # Get coefficients for manual IIR bandpass filter
    b_manual, a_manual = butterworth_bp_manual(order, low_cutoff, high_cutoff, fs)
    
    # Get coefficients for optimized IIR bandpass filter
    b_opt, a_opt = butterworth_bp_manual_opt(order, low_cutoff, high_cutoff, fs)

    # Compute frequency response for manual filter
    w_manual, h_manual = freqz(b_manual, a_manual, worN=8000)
    # Compute frequency response for optimized filter
    w_opt, h_opt = freqz(b_opt, a_opt, worN=8000)

    # Plot frequency responses
    plt.figure()
    plt.plot(w_manual * fs / (2 * np.pi), 20 * np.log10(np.abs(h_manual) + 1e-10), 'b', label='Manual')
    plt.plot(w_opt * fs / (2 * np.pi), 20 * np.log10(np.abs(h_opt) + 1e-10), 'r--', label='Optimized')
    plt.title('Frequency Response of IIR Bandpass Filter - Optimized')
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
    >>> plot_bandpass_filter_opt_coefficients(1000, 100, 500, 8000)
    """
    validate_inputs(order, low_cutoff, high_cutoff, fs)

    # Get coefficients for manual IIR bandpass filter
    b_manual, a_manual = butterworth_bp_manual(order, low_cutoff, high_cutoff, fs)
    
    # Get coefficients for optimized IIR bandpass filter
    b_opt, a_opt = butterworth_bp_manual_opt(order, low_cutoff, high_cutoff, fs)

    # Plot coefficients
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(b_manual)
    plt.title('Manual Optimized IIR Bandpass Filter Coefficients (Numerator)')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.stem(b_opt)
    plt.title('Optimized IIR Bandpass Filter Coefficients (Numerator)')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

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

    # Get coefficients for manual IIR bandpass filter
    b_manual, a_manual = butterworth_bp_manual(order, low_cutoff, high_cutoff, fs)

    # Plot coefficients
    plt.figure(figsize=(12, 6))

    # Plot numerator coefficients
    plt.subplot(1, 2, 1)
    plt.stem(b_manual)
    plt.title('Manual IIR Bandpass Filter Coefficients (Numerator)')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    # Plot denominator coefficients
    plt.subplot(1, 2, 2)
    plt.stem(a_manual)
    plt.title('Manual IIR Bandpass Filter Coefficients (Denominator)')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.tight_layout()
    plt.show()
