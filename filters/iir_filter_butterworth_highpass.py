import numpy as np
from scipy.signal import butter, freqz, bilinear
import matplotlib.pyplot as plt

class FilterErrorHp(Exception):
    """Custom exception for filter errors."""
    pass

def validate_inputs(order, cutoff, fs):
    if not isinstance(order, int) or order <= 0:
        raise FilterErrorHp("Order must be a positive integer.")
    if not isinstance(cutoff, (int, float)) or cutoff <= 0:
        raise FilterErrorHp("Cutoff frequency must be a positive number.")
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise FilterErrorHp("Sampling frequency must be a positive number.")
    if cutoff >= fs / 2:
        raise FilterErrorHp("Cutoff frequency must be less than half the sampling frequency.")

def butterworth_hp_manual(order: int, cutoff: float, fs: float) -> tuple:
    """
    Generates Butterworth high-pass filter coefficients manually with proper normalization.

    Parameters:
    order (int): The order of the filter.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency.

    Returns:
    tuple: Numerator (b) and denominator (a) coefficients of the filter.

    Example:
    >>> order = 4
    >>> cutoff = 1000.0  # 1 kHz
    >>> fs = 8000.0  # 8 kHz sampling frequency
    >>> b, a = butterworth_hp_manual(order, cutoff, fs)
    >>> print(b)
    >>> print(a)
    """
    validate_inputs(order, cutoff, fs)
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    # Analog cutoff frequency (rad/s)
    wc = 2 * np.pi * cutoff  # Analog cutoff frequency

    # Compute analog poles for Butterworth filter
    poles = [
        wc * np.exp(1j * np.pi * (0.38 * k + 1) / (0.38 * order)) for k in range(order)
    ]
    poles = np.array([p for p in poles if np.real(p) < 0])  # Keep only left half-plane poles

    # Analog filter coefficients
    a_analog = np.poly(poles)
    b_analog = np.array([1.0] + [0] * order)  # Numerator for high-pass (reverse frequency response)

    # Bilinear transformation to digital filter
    b, a = bilinear(b_analog, a_analog, fs)
    return b, a

def butterworth_hp_builtin(order: int, cutoff: float, fs: float) -> tuple:
    """
    Generates Butterworth high-pass filter coefficients using built-in function.

    Parameters:
    order (int): The order of the filter.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency.

    Returns:
    tuple: Numerator (b) and denominator (a) polynomials of the IIR filter.

    Example:
    >>> order = 4
    >>> cutoff = 1000.0  # 1 kHz
    >>> fs = 8000.0  # 8 kHz sampling frequency
    >>> b, a = butterworth_hp_builtin(order, cutoff, fs)
    >>> print(b)
    >>> print(a)
    """
    validate_inputs(order, cutoff, fs)
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return b, a

def plot_coefficients(order: int, cutoff: float, fs: float) -> None:
    """
    Plots the coefficients of manually generated and built-in Butterworth high-pass filters.
    """
    validate_inputs(order, cutoff, fs)
    b_manual, a_manual = butterworth_hp_manual(order, cutoff, fs)
    b_builtin, a_builtin = butterworth_hp_builtin(order, cutoff, fs)

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

def plot_frequency_response(order: int, cutoff: float, fs: float) -> None:
    """
    Plots the frequency response of manually generated and built-in Butterworth high-pass filters.
    """
    validate_inputs(order, cutoff, fs)
    b_manual, a_manual = butterworth_hp_manual(order, cutoff, fs)
    b_builtin, a_builtin = butterworth_hp_builtin(order, cutoff, fs)

    w_manual, h_manual = freqz(b_manual, a_manual, worN=8000)
    w_builtin, h_builtin = freqz(b_builtin, a_builtin, worN=8000)

    plt.figure()
    plt.plot(w_manual * fs / (2 * np.pi), 20 * np.log10(np.abs(h_manual)), 'b', label='Manual')
    plt.plot(w_builtin * fs / (2 * np.pi), 20 * np.log10(np.abs(h_builtin)), 'g--', label='Built-in')
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid()
    plt.show()

def compare_coefficients(order: int, cutoff: float, fs: float) -> None:
    """
    Compares the coefficients of manually generated and built-in Butterworth high-pass filters.
    """
    validate_inputs(order, cutoff, fs)
    b_manual, a_manual = butterworth_hp_manual(order, cutoff, fs)
    b_builtin, a_builtin = butterworth_hp_builtin(order, cutoff, fs)

    print("Manual Filter Coefficients:")
    print("Numerator (b):", b_manual)
    print("Denominator (a):", a_manual)

    print("\nBuilt-in Filter Coefficients:")
    print("Numerator (b):", b_builtin)
    print("Denominator (a):", a_builtin)

# Filter parameters
fs = 1000  # Sampling frequency
cutoff = 100  # Cutoff frequency
order = 4  # Filter order

# Plotting coefficients
#plot_coefficients(order, cutoff, fs)

# Plotting frequency response
#plot_frequency_response(order, cutoff, fs)

# Comparing coefficients
#compare_coefficients(order, cutoff, fs)
