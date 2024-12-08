import pytest
import numpy as np
from scipy.signal import lfilter, iirfilter
from filters.iir_filter_butterworth_highpass import butterworth_hp_manual, butterworth_hp_builtin, FilterErrorHp
from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual, butterworth_lp_builtin, FilterErrorLp

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

# Additional tests for lowpass IIR filter
def test_lowpass_iir_filter_manual_valid():
    """
    Test lowpass IIR filter with valid parameters.
    """
    order = 4
    cutoff_freq = 100
    sample_rate = 1000
    b, a = butterworth_lp_manual(order, cutoff_freq, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

def test_lowpass_iir_filter_builtin_valid():
    """
    Test lowpass IIR filter using built-in function with valid parameters.
    """
    order = 4
    cutoff_freq = 100
    sample_rate = 1000
    b, a = butterworth_lp_builtin(order, cutoff_freq, sample_rate)
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
    cutoff_freq = 100
    sample_rate = 1000
    b, a = butterworth_hp_manual(order, cutoff_freq, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == order + 1
    assert len(a) == order + 1

def test_highpass_iir_filter_builtin_valid():
    """
    Test highpass IIR filter using built-in function with valid parameters.
    """
    order = 4
    cutoff_freq = 100
    sample_rate = 1000
    b, a = butterworth_hp_builtin(order, cutoff_freq, sample_rate)
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
