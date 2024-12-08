import numpy as np
from scipy.signal import bilinear, freqz
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import bilinear, freqz
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import bilinear, freqz
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import bilinear, freqz
import matplotlib.pyplot as plt

def butterworth_bp_manual(order: int, low_cutoff: float, high_cutoff: float, fs: float) -> tuple:
    """
    Manually generates Butterworth band-pass filter coefficients using bilinear transformation.

    Parameters:
    order (int): The order of the filter.
    low_cutoff (float): The lower cutoff frequency of the filter.
    high_cutoff (float): The upper cutoff frequency of the filter.
    fs (float): The sampling frequency.

    Returns:
    tuple: Numerator (b) and denominator (a) coefficients of the filter.
    """
    nyquist = fs / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    if low >= high:
        raise ValueError("Low cutoff frequency must be less than high cutoff frequency.")

    # Prewarp frequencies for bilinear transform
    prewarp_low = 2 * np.pi * low_cutoff
    prewarp_high = 2 * np.pi * high_cutoff

    # Center frequency and bandwidth in the analog domain
    w0 = np.sqrt(prewarp_low * prewarp_high)  # Analog center frequency
    bw = prewarp_high - prewarp_low  # Bandwidth

    # Analog lowpass prototype poles
    poles = [
        np.exp(1j * np.pi * (0.5 * k + 1) / (0.5 * order))
        for k in range(order)
    ]
    poles = np.array([p for p in poles if np.real(p) < 0])  # Left-half plane poles

    # Transform lowpass poles to bandpass poles
    analog_poles = []
    for p in poles:
        analog_poles.append(bw / 0.5 * p + 1j * w0 / 0.5)
        analog_poles.append(bw / 0.5 * p - 1j * w0 / 0.5)

    # Coefficients of the analog band-pass filter
    a_analog = np.poly(analog_poles)
    b_analog = np.array([bw**order])

    # Normalize to ensure gain at the center frequency is 1
    w0_digital = 2 * fs * np.tan(w0 / (2 * fs))
    gain = np.abs(np.polyval(b_analog, 1j * w0_digital) / np.polyval(a_analog, 1j * w0_digital))
    b_analog = b_analog / gain

    # Bilinear transformation
    b, a = bilinear(b_analog, a_analog, fs)
    return b, a


def butterworth_bp_builtin(order: int, low_cutoff: float, high_cutoff: float, fs: float) -> tuple:
    """
    Generates Butterworth band-pass filter coefficients using built-in functions.

    Parameters:
    order (int): The order of the filter.
    low_cutoff (float): The lower cutoff frequency of the filter.
    high_cutoff (float): The upper cutoff frequency of the filter.
    fs (float): The sampling frequency.

    Returns:
    tuple: Numerator (b) and denominator (a) coefficients of the filter.
    """
    from scipy.signal import butter
    b, a = butter(order, [low_cutoff / (fs / 2), high_cutoff / (fs / 2)], btype='band')
    return b, a

# Plotting functions remain unchanged
def plot_frequency_response(b_manual, a_manual, b_builtin, a_builtin, fs: float) -> None:
    """
    Plots the frequency response of manually generated and built-in Butterworth band-pass filters.

    Parameters:
    b_manual (array): Numerator coefficients of the manual filter.
    a_manual (array): Denominator coefficients of the manual filter.
    b_builtin (array): Numerator coefficients of the built-in filter.
    a_builtin (array): Denominator coefficients of the built-in filter.
    fs (float): Sampling frequency.
    """
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


#if __name__ == "__main__":
    fs = 1000  # Sampling frequency
    order = 4  # Filter order

    # Band-pass filter parameters
    low_cutoff_bp = 100  # Lower cutoff frequency (Hz)
    high_cutoff_bp = 300  # Upper cutoff frequency (Hz)

    # Generate coefficients
    b_manual_bp, a_manual_bp = butterworth_bp_manual(order, low_cutoff_bp, high_cutoff_bp, fs)
    b_builtin_bp, a_builtin_bp = butterworth_bp_builtin(order, low_cutoff_bp, high_cutoff_bp, fs)

    # Plot frequency response
    plot_frequency_response(b_manual_bp, a_manual_bp, b_builtin_bp, a_builtin_bp, fs)
