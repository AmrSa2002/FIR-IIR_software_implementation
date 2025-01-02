import numpy as np
from scipy.signal import butter, freqz, bilinear
import matplotlib.pyplot as plt
#from iir_compare import measure_performance_highpass_iir

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

def butterworth_hp_manual_opt(order: int, cutoff: float, fs: float) -> tuple:
    """
    Generates optimized Butterworth high-pass filter coefficients with proper normalization using vectorized operations.

    Parameters:
    order (int): The order of the filter.
    cutoff (float): The cutoff frequency of the filter (in Hz).
    fs (float): The sampling frequency (in Hz).

    Returns:
    tuple: Numerator (b) and denominator (a) coefficients of the filter.
    
    Example:
    >>> order = 4
    >>> cutoff = 1000.0  # 1 kHz
    >>> fs = 8000.0  # 8 kHz sampling frequency
    >>> b, a = butterworth_hp_manual_opt(order, cutoff, fs)
    >>> print(b)
    >>> print(a)
    """
    # Validate inputs
    validate_inputs(order, cutoff, fs)

    # Nyquist frequency and normalized cutoff
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    # Precompute the frequency (analog) for each pole
    k = np.arange(1, order + 1, dtype=np.float32)
    angles = (2 * k - 1) * np.pi / (2 * order)
    
    # Calculate poles using a vectorized approach
    poles = np.exp(1j * angles)
    poles = np.multiply(poles, normalized_cutoff)

    # Keep only the left half-plane poles (for stability in digital filter)
    poles = poles[np.real(poles) < 0]

    # Analog filter coefficients
    a_analog = np.poly(poles)  # Analog polynomial coefficients for denominator
    b_analog = np.array([1.0] + [0] * order)  # Numerator for high-pass filter (reverse frequency response)

    # Apply bilinear transformation to get digital filter coefficients
    b, a = bilinear(b_analog, a_analog, fs)

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
    Plots the frequency response of manually generated, optimized and built-in Butterworth high-pass filters.
    """
    # Validate inputs
    validate_inputs(order, cutoff, fs)

    # Get coefficients for manually generated filter
    b_manual, a_manual = butterworth_hp_manual(order, cutoff, fs)

    # Get coefficients for optimized filter
    b_opt, a_opt = butterworth_hp_manual_opt(order, cutoff, fs)

    # Get coefficients for built-in filter
    b_builtin, a_builtin = butterworth_hp_builtin(order, cutoff, fs)

    # Compute frequency responses for each filter
    w_manual, h_manual = freqz(b_manual, a_manual, worN=8000)
    w_opt, h_opt = freqz(b_opt, a_opt, worN=8000)
    w_builtin, h_builtin = freqz(b_builtin, a_builtin, worN=8000)

    # Add a small value epsilon to avoid log errors when logging zeros
    eps = 1e-10

    # Plot the frequency responses
    plt.figure()
    plt.plot(w_manual * fs / (2 * np.pi), 20 * np.log10(np.abs(h_manual) + eps), 'b', label='Manual')
    plt.plot(w_opt * fs / (2 * np.pi), 20 * np.log10(np.abs(h_opt) + eps), 'r--', label='Optimized')
    plt.plot(w_builtin * fs / (2 * np.pi), 20 * np.log10(np.abs(h_builtin) + eps), 'g--', label='Built-in')

    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid()
    plt.show()

#def compare_coefficients(order: int, cutoff: float, fs: float) -> None:
  #  """
  #  Compares the coefficients of manually generated and built-in Butterworth high-pass filters.
   # """
   # validate_inputs(order, cutoff, fs)
   # b_manual, a_manual = butterworth_hp_manual(order, cutoff, fs)
  #  b_builtin, a_builtin = butterworth_hp_builtin(order, cutoff, fs)

   # print("Manual Filter Coefficients:")
   # print("Numerator (b):", b_manual)
    #print("Denominator (a):", a_manual)

   # print("\nBuilt-in Filter Coefficients:")
   # print("Numerator (b):", b_builtin)
   # print("Denominator (a):", a_builtin)

# Filter parameters
fs = 1000  # Sampling frequency
cutoff = 100  # Cutoff frequency
order = 4  # Filter order




def plot_iir_highpass_filter_opt_responses(order: int, cutoff: float, fs: int):
    """
    Plots the frequency responses of the high-pass filters generated by both manual and optimized IIR methods.

    Parameters:
    cutoff (float): The cutoff frequency of the filter (in Hz).
    order (int): The order of the filter.
    fs (int): The sample rate of the signal (in Hz).
    
    Raises:
    ValueError: If order is not a positive integer or if cutoff is not a positive value.
    
    Example:
    --------
    >>> plot_highpass_filter_opt_responses(1000, 4, 8000)
    """
    validate_inputs(order, cutoff, fs)

    # Get coefficients for manual IIR filter
    b_manual, a_manual = butterworth_hp_manual(order, cutoff, fs)
    
    # Get coefficients for optimized IIR filter
    b_opt, a_opt = butterworth_hp_manual_opt(order, cutoff, fs)

    # Compute frequency response for manual filter
    w_manual, h_manual = freqz(b_manual, a_manual, worN=8000)
    # Compute frequency response for optimized filter
    w_opt, h_opt = freqz(b_opt, a_opt, worN=8000)

    # Plot frequency responses
    plt.figure()
    plt.plot(w_manual * fs / (2 * np.pi), 20 * np.log10(np.abs(h_manual) + 1e-10), 'b', label='Manual')
    plt.plot(w_opt * fs / (2 * np.pi), 20 * np.log10(np.abs(h_opt) + 1e-10), 'r--', label='Optimized')
    plt.title('Frequency Response of IIR High-pass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid()
    plt.show()



def plot_iir_highpass_filter_opt_coefficients(order: int, cutoff: float,  fs: int):
    """
    Plots the high-pass filter coefficients generated by both manual and optimized IIR methods side by side.

    Parameters:
    cutoff (float): The cutoff frequency of the filter (in Hz).
    order (int): The order of the filter.
    fs (int): The sample rate of the signal (in Hz).
    
    Raises:
    ValueError: If order is not a positive integer or if cutoff is not a positive value.
    
    Example:
    --------
    >>> plot_highpass_filter_opt_coefficients(1000, 4, 8000)
    """
    validate_inputs(order, cutoff, fs)

    # Get coefficients for manual IIR filter
    b_manual, a_manual = butterworth_hp_manual(order, cutoff, fs)
    
    # Get coefficients for optimized IIR filter
    b_opt, a_opt = butterworth_hp_manual_opt(order, cutoff, fs)

    # Plot coefficients
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(b_manual)
    plt.title('Manual Optimized IIR Highpass Filter Coefficients (Numerator)')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.stem(b_opt)
    plt.title('Optimized IIR Highpass Filter Coefficients (Numerator)')
    plt.xlabel('Tap')
    plt.ylabel('Coefficient Value')
    plt.grid()

    plt.tight_layout()
    plt.show()




