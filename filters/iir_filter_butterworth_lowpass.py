import numpy as np
from scipy.signal import butter, freqz, bilinear
import matplotlib.pyplot as plt

class FilterErrorLp(Exception):
    """Custom exception for filter errors."""
    pass

def validate_inputs(order, cutoff, fs):
    if not isinstance(order, int) or order <= 0:
        raise FilterErrorLp("Order must be a positive integer.")
    if not isinstance(cutoff, (int, float)) or cutoff <= 0:
        raise FilterErrorLp("Cutoff frequency must be a positive number.")
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise FilterErrorLp("Sampling frequency must be a positive number.")
    if cutoff >= fs / 2:
        raise FilterErrorLp("Cutoff frequency must be less than half the sampling frequency.")

def butterworth_lp_manual(order: int, cutoff: float, fs: float) -> tuple:
    validate_inputs(order, cutoff, fs)
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    # Analog cutoff frequency (rad/s)
    wc = 2 * np.pi * cutoff

    # Compute analog poles for Butterworth filter
    poles = [
        wc * np.exp(1j * np.pi * (0.38 * k + 1) / (0.38 * order)) for k in range(order)
    ]
    poles = np.array([p for p in poles if np.real(p) < 0])

    # Analog filter coefficients
    a_analog = np.poly(poles)
    b_analog = np.array([wc**order])

    # Bilinear transformation to digital filter
    b, a = bilinear(b_analog, a_analog, fs)
    return b, a

def butterworth_lp_builtin(order: int, cutoff: float, fs: float) -> tuple:
    validate_inputs(order, cutoff, fs)
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    return b, a

def butterworth_lp_manual_opt(order: int, cutoff: float, fs: float) -> tuple:
    validate_inputs(order, cutoff, fs)

    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    wc = 2 * np.pi * cutoff

    # Precompute constant factor for poles to avoid repetitive computation in the loop
    angle_factor = np.pi * (0.38 * np.arange(order) + 1) / (0.38 * order)

    # Analog poles for Butterworth filter, apply condition to keep only stable poles
    poles = wc * np.exp(1j * angle_factor)
    poles = poles[np.real(poles) < 0]

    # Analog filter coefficients using numpy's poly function
    a_analog = np.poly(poles)
    b_analog = np.array([wc**order])

    # Bilinear transformation to digital filter
    b, a = bilinear(b_analog, a_analog, fs)
    return b, a


def plot_coefficients(order: int, cutoff: float, fs: float) -> None:
    validate_inputs(order, cutoff, fs)
    b_manual, a_manual = butterworth_lp_manual(order, cutoff, fs)
    b_builtin, a_builtin = butterworth_lp_builtin(order, cutoff, fs)

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

def plot_opt_coefficients(order: int, cutoff: float, fs: float) -> None:
    validate_inputs(order, cutoff, fs)
    b_manual, a_manual = butterworth_lp_manual_opt(order, cutoff, fs)
    b_builtin, a_builtin = butterworth_lp_builtin(order, cutoff, fs)

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

def plot_lowpass_filter_responses(order: int, cutoff: float, fs: float) -> None:
    validate_inputs(order, cutoff, fs)
    b_manual, a_manual = butterworth_lp_manual(order, cutoff, fs)
    b_opt, a_opt = butterworth_lp_manual_opt(order, cutoff, fs)
    b_builtin, a_builtin = butterworth_lp_builtin(order, cutoff, fs)

    w_manual, h_manual = freqz(b_manual, a_manual, worN=8000)
    w_opt, h_opt = freqz(b_opt, a_opt, worN=8000)
    w_builtin, h_builtin = freqz(b_builtin, a_builtin, worN=8000)

    eps = 1e-10

    plt.figure()
    plt.plot(w_manual * fs / (2 * np.pi), 20 * np.log10(np.abs(h_manual) + eps), 'b', label='Manual')
    plt.plot(w_opt * fs / (2 * np.pi), 20 * np.log10(np.abs(h_opt) + eps), 'r--', label='Optimized')
    plt.plot(w_builtin * fs / (2 * np.pi), 20 * np.log10(np.abs(h_builtin) + eps), 'g--', label='Built-in')

    plt.title('Frequency Response IIR Lowpass filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid()
    plt.show()

#def compare_coefficients(order: int, cutoff: float, fs: float) -> None:
    #validate_inputs(order, cutoff, fs)
    #b_manual, a_manual = butterworth_lp_manual(order, cutoff, fs)
    #b_builtin, a_builtin = butterworth_lp_builtin(order, cutoff, fs)

   # print("Manual Filter Coefficients:")
   # print("Numerator (b):", b_manual)
   # print("Denominator (a):", a_manual)

   # print("\nBuilt-in Filter Coefficients:")
    #print("Numerator (b):", b_builtin)
   # print("Denominator (a):", a_builtin)

def plot_lowpass_filter_opt_responses(order: int, cutoff: float, fs: int):
    validate_inputs(order, cutoff, fs)

    b_manual, a_manual = butterworth_lp_manual(order, cutoff, fs)
    b_opt, a_opt = butterworth_lp_manual_opt(order, cutoff, fs)

    w_manual, h_manual = freqz(b_manual, a_manual, worN=8000)
    w_opt, h_opt = freqz(b_opt, a_opt, worN=8000)

    plt.figure()
    plt.plot(w_manual * fs / (2 * np.pi), 20 * np.log10(np.abs(h_manual) + 1e-10), 'b', label='Manual')
    plt.plot(w_opt * fs / (2 * np.pi), 20 * np.log10(np.abs(h_opt) + 1e-10), 'r--', label='Optimized')
    plt.title('Frequency Response of IIR Low-pass Filter - Optimized')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.grid()
    plt.show()
