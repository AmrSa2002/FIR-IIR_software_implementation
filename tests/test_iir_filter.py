import pytest
import numpy as np
from scipy.signal import lfilter, iirfilter
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from filters.iir_filter_butterworth_highpass import butterworth_hp_manual, butterworth_hp_builtin, butterworth_hp_manual_opt, FilterErrorHp
from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual, butterworth_lp_builtin, butterworth_lp_manual_opt, plot_frequency_response, FilterErrorLp
from filters.iir_filter_butterworth_bandpass import butterworth_bp_manual, butterworth_bp_builtin, butterworth_bp_manual_opt, FilterErrorBp

# Define the IIR filter parameters
def create_iir_filter(filter_type, order, cutoff, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = iirfilter(order, normal_cutoff, btype=filter_type, ftype='butter')
    return b, a

# Test the lowpass filter
def test_lowpass_filter():
    fs = 1000  # Sampling frequency
    cutoff = 100  # Desired cutoff frequency of the filter, Hz
    order = 4  # Filter order
    b, a = create_iir_filter('low', order, cutoff, fs)

    # Generate a signal for testing
    t = np.linspace(0, 1.0, fs)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    # Apply the filter
    filtered_signal = lfilter(b, a, signal)

    # Check if the high frequency component is attenuated
    assert np.max(np.abs(filtered_signal)) < np.max(np.abs(signal))

# Test the highpass filter
def test_highpass_filter():
    fs = 1000  # Sampling frequency
    cutoff = 100  # Desired cutoff frequency of the filter, Hz
    order = 4  # Filter order
    b, a = create_iir_filter('high', order, cutoff, fs)

    # Generate a signal for testing
    t = np.linspace(0, 1.0, fs)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    # Apply the filter
    filtered_signal = lfilter(b, a, signal)

    # Check if the low frequency component is attenuated
    assert np.max(np.abs(filtered_signal)) < np.max(np.abs(signal))

# Test invalid inputs for lowpass filter
def test_invalid_inputs_lowpass():
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual(-1, 100, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual(4, -100, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual(4, 100, -1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual(4, 1000, 1000)
        
# Test invalid inputs for lowpass filter optimized
def test_invalid_inputs_lowpass_opt():
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual_opt(-1, 100, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual_opt(4, -100, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual_opt(4, 100, -1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual_opt(4, 1000, 1000)

# Test invalid inputs for highpass filter
def test_invalid_inputs_highpass():
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual(-1, 100, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual(4, -100, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual(4, 100, -1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual(4, 1000, 1000)

# Test invalid inputs for highpass filter optimized
def test_invalid_inputs_highpass_opt():
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual_opt(-1, 100, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual_opt(4, -100, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual_opt(4, 100, -1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual_opt(4, 1000, 1000)

# Additional tests for lowpass IIR filter
def test_lowpass_iir_filter_manual_valid():
    """
    Test lowpass IIR filter with valid parameters.
    """
    order = 4
    cutoff = 100
    sample_rate = 1000
    b, a = butterworth_lp_manual(order, cutoff, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

def test_lowpass_iir_filter_manual_opt_valid():
    """
    Test lowpass IIR filter with valid parameters.
    """
    order = 4
    cutoff = 100
    sample_rate = 1000
    b, a = butterworth_lp_manual_opt(order, cutoff, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

def test_lowpass_iir_filter_builtin_valid():
    """
    Test lowpass IIR filter using built-in function with valid parameters.
    """
    order = 4
    cutoff= 100
    sample_rate = 1000
    b, a = butterworth_lp_builtin(order, cutoff, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

# Additional tests for highpass IIR filter
def test_highpass_iir_filter_manual_valid():
    """
    Test highpass IIR filter with valid parameters.
    """
    order = 4
    cutoff = 100
    sample_rate = 1000
    b, a = butterworth_hp_manual(order, cutoff, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

def test_highpass_iir_filter_manual_opt_valid():
    """
    Test highpass IIR filter with valid parameters.
    """
    order = 4
    cutoff = 100
    sample_rate = 1000
    b, a = butterworth_hp_manual_opt(order, cutoff, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

def test_highpass_iir_filter_builtin_valid():
    """
    Test highpass IIR filter using built-in function with valid parameters.
    """
    order = 4
    cutoff = 100
    sample_rate = 1000
    b, a = butterworth_hp_builtin(order, cutoff, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

# Test edge cases for lowpass IIR filter
def test_lowpass_iir_filter_manual_edge_cases():
    """
    Test lowpass IIR filter with edge case parameters.
    """
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual(0, 100, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual(4, 0, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual(4, 100, 0)

def test_lowpass_iir_filter_manual_opt_edge_cases():
    """
    Test lowpass IIR filter with edge case parameters.
    """
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual_opt(0, 100, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual_opt(4, 0, 1000)
    with pytest.raises(FilterErrorLp):
        butterworth_lp_manual_opt(4, 100, 0)

# Test edge cases for highpass IIR filter
def test_highpass_iir_filter_manual_edge_cases():
    """
    Test highpass IIR filter with edge case parameters.
    """
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual(0, 100, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual(4, 0, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual(4, 100, 0)

def test_highpass_iir_filter_manual_opt_edge_cases():
    """
    Test highpass IIR filter with edge case parameters.
    """
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual_opt(0, 100, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual_opt(4, 0, 1000)
    with pytest.raises(FilterErrorHp):
        butterworth_hp_manual_opt(4, 100, 0)

#TESTS FOR BANDPASS

# Test valid parameters for manual bandpass filter
def test_bandpass_iir_filter_manual_valid():
    """
    Test bandpass IIR filter with valid parameters.
    """
    order = 4
    lowcut = 50
    highcut = 150
    sample_rate = 500
    b, a = butterworth_bp_manual(lowcut, highcut, sample_rate, order)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == len(a)

# Test valid parameters for optimized manual bandpass filter
def test_bandpass_iir_filter_manual_opt_valid():
    """
    Test bandpass IIR filter with valid parameters.
    """
    order = 4
    lowcut = 50
    highcut = 150
    sample_rate = 500
    b, a = butterworth_bp_manual_opt(lowcut, highcut, sample_rate, order)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == len(a)

# Test valid parameters for built-in bandpass filter
def test_bandpass_iir_filter_builtin_valid():
    """
    Test bandpass IIR filter using built-in function with valid parameters.
    """
    order = 4
    lowcut = 50
    highcut = 150
    sample_rate = 500
    b, a = butterworth_bp_builtin(lowcut, highcut, sample_rate, order)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == len(a)

# Test invalid parameters for bandpass filter
def test_invalid_inputs_bandpass():
    """
    Test invalid input scenarios for bandpass filter.
    """
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(-50, 150, 500, 4)  # Negative lowcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(50, -150, 500, 4)  # Negative highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(50, 150, -500, 4)  # Negative sample rate
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(50, 500, 500, 4)  # Highcut >= Nyquist

# Test edge cases for bandpass filter
def test_bandpass_iir_filter_manual_edge_cases():
    """
    Test edge cases for bandpass IIR filter.
    """
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(0, 150, 500, 4)  # Zero lowcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(50, 0, 500, 4)  # Zero highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(50, 150, 0, 4)  # Zero sample rate

# Test optimized filter edge cases
def test_bandpass_iir_filter_manual_opt_edge_cases():
    """
    Test edge cases for optimized bandpass IIR filter.
    """
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(0, 150, 500, 4)  # Zero lowcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(50, 0, 500, 4)  # Zero highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(50, 150, 0, 4)  # Zero sample rate

# Test built-in filter edge cases
def test_bandpass_iir_filter_builtin_edge_cases():
    """
    Test edge cases for built-in bandpass IIR filter.
    """
    with pytest.raises(ValueError):  # Built-in scipy function raises ValueError
        butterworth_bp_builtin(0, 150, 500, 4)  # Zero lowcut
    with pytest.raises(ValueError):
        butterworth_bp_builtin(50, 0, 500, 4)  # Zero highcut
    with pytest.raises(ValueError):
        butterworth_bp_builtin(50, 150, 0, 4)  # Zero sample rate


# Test plotting functions (these tests will not check the plots visually but will ensure no errors are raised)
def test_plot_lowpass_iir_filter_responses():
    """
    Test the plotting of lowpass filter responses.
    """
    from filters.iir_filter_butterworth_lowpass import plot_frequency_response
    plot_frequency_response(order=4, cutoff=0.3, fs=1000)

def test_plot_lowpass_iir_opt_filter_responses():
    """
    Test the plotting of lowpass filter responses.
    """
    from filters.iir_filter_butterworth_lowpass import plot_lowpass_filter_opt_responses
    plot_lowpass_filter_opt_responses(order = 4, cutoff=0.3, fs=1000)

def test_plot_highpass_iir_opt_filter_responses():
    """
    Test the plotting of optimized highpass filter responses.
    """
    from filters.iir_filter_butterworth_highpass import plot_highpass_filter_opt_responses
    plot_highpass_filter_opt_responses(order = 4, cutoff=0.3, fs=1000)

def test_plot_highpass_iir_filter_responses():
    """
    Test the plotting of highpass filter responses.
    """
    from filters.iir_filter_butterworth_highpass import plot_frequency_response
    plot_frequency_response(order = 4, cutoff=0.3, fs=1000)

def test_plot_bandpass_iir_filter_responses():
    """
    Test the plotting of bandpass filter responses.
    """
    from filters.iir_filter_butterworth_bandpass import plot_frequency_response
    plot_frequency_response(b = 4, a = 3, lowcut=0.2, highcut=0.4, fs=1000)

def test_plot_bandpass_iir_filter_opt_responses():
    """
    Test the plotting of bandpass filter responses.
    """
    from filters.iir_filter_butterworth_bandpass import plot_bandpass_filter_opt_responses
    plot_bandpass_filter_opt_responses(b = 4, a = 3, lowcut=0.2, highcut=0.4, fs=1000)

def test_plot_lowpass_iir_filter_coefficients():
    """
    Test the plotting of lowpass filter coefficients.
    """
    from filters.iir_filter_butterworth_lowpass import plot_coefficients
    plot_coefficients(order = 4, cutoff=0.3, fs = 1000)

def test_plot_lowpass_filter_opt_coefficients():
    """
    Test the plotting of lowpass filter coefficients.
    """
    from filters.iir_filter_butterworth_lowpass import plot_opt_coefficients
    plot_opt_coefficients(order = 4, cutoff=0.3, fs=1000)

def test_plot_highpass_iir_filter_coefficients():
    """
    Test the plotting of highpass filter coefficients.
    """
    from filters.iir_filter_butterworth_highpass import plot_coefficients
    plot_coefficients(order = 4, cutoff=0.3, fs = 1000)


def test_plot_highpass_iir_filter_opt_coefficients():
    """
    Test the plotting of optimized highpass filter coefficients.
    """
    from filters.iir_filter_butterworth_highpass import plot_highpass_filter_opt_coefficients
    plot_highpass_filter_opt_coefficients(order = 4, cutoff = 0.3, fs = 1000)

def test_plot_bandpass_iir_opt_filter_coefficients():
    """
    Test the plotting of bandpass filter coefficients.
    """
    from filters.iir_filter_butterworth_bandpass import plot_bandpass_filter_opt_coefficients
    plot_bandpass_filter_opt_coefficients(order = 4, low_cutoff=0.2, high_cutoff=0.4, fs = 1000)

def test_plot_bandpass_iir_filter_coefficients():
    """
    Test the plotting of optimized IIR bandpass filter coefficients.
    """
    from filters.iir_filter_butterworth_bandpass import plot_bandpass_filter_coefficients
    plot_bandpass_filter_coefficients(order = 4, low_cutoff=0.2, high_cutoff=0.4, fs = 1000)