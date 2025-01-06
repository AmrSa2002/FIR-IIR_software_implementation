import pytest
import numpy as np
from scipy.signal import lfilter, iirfilter
import sys
import os
from scipy.signal import tf2zpk, zpk2tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from filters.iir_filter_butterworth_highpass import butterworth_hp_manual, butterworth_hp_builtin, butterworth_hp_manual_opt, FilterErrorHp
from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual, butterworth_lp_builtin, butterworth_lp_manual_opt, FilterErrorLp
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
    assert len(b) == order - 1
    assert len(a) == order - 1

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

    assert lowcut > 0, "Lowcut frequency must be greater than zero"
    assert highcut > 0, "Highcut frequency must be greater than zero"
    assert highcut < sample_rate / 2, "Highcut frequency must be less than half the sample rate"
    b, a = butterworth_bp_manual_opt(order,lowcut, highcut, sample_rate)
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
    b, a = butterworth_bp_builtin(order,lowcut, highcut, sample_rate)
    assert isinstance(b, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert len(b) == len(a)

# Test invalid parameters for bandpass filter
def test_invalid_inputs_bandpass():
    """
    Test invalid input scenarios for bandpass filter.
    """
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4,-50, 150, 500)  # Negative lowcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 50, -150, 500)  # Negative highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 50, 150, -500)  # Negative sample rate
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 50, 500, 500)  # Highcut >= Nyquist
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 200, 150, 500)  # Lowcut >= Highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 500, 150, 500)  # Lowcut >= Nyquist
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(-4, 50, 500, 500)  # Negative order
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 50, 500, 0)  # Sampling frequency zero
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 0, 500, 500)  # Lowcut zero
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 50, 0, 500)  # Highcut zero
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4,-50, 150, 500)  # Negative lowcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 50, -150, 500)  # Negative highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 50, 150, -500)  # Negative sample rate
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 50, 500, 500)  # Highcut >= Nyquist
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 200, 150, 500)  # Lowcut >= Highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 500, 150, 500)  # Lowcut >= Nyquist
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(-4, 50, 500, 500)  # Negative order
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 50, 500, 0)  # Sampling frequency zero
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 0, 500, 500)  # Lowcut zero
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 50, 0, 500)  # Highcut zero

# Test edge cases for bandpass filter
def test_bandpass_iir_filter_manual_edge_cases():
    """
    Test edge cases for bandpass IIR filter.
    """
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 0, 150, 500)  # Zero lowcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 50, 0, 500)  # Zero highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual(4, 50, 150, 0)  # Zero sample rate

# Test optimized filter edge cases
def test_bandpass_iir_filter_manual_opt_edge_cases():
    """
    Test edge cases for optimized bandpass IIR filter.
    """
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 0, 150, 500)  # Zero lowcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 50, 0, 500)  # Zero highcut
    with pytest.raises(FilterErrorBp):
        butterworth_bp_manual_opt(4, 50, 150, 500)  # Zero sample rate

# Test built-in filter edge cases
def test_bandpass_iir_filter_builtin_edge_cases():
    """
    Test edge cases for built-in bandpass IIR filter.
    """
    with pytest.raises(ValueError):  # Built-in scipy function raises ValueError
        butterworth_bp_builtin(4, 0, 150, 500)  # Zero lowcut
    with pytest.raises(ValueError):
        butterworth_bp_builtin(4, 50, 0, 500)  # Zero highcut
    with pytest.raises(ValueError):
        butterworth_bp_builtin(4, 50, 150, 0)  # Zero sample rate


# Test plotting functions (these tests will not check the plots visually but will ensure no errors are raised)
def test_plot_lowpass_iir_filter_responses():
    """
    Test the plotting of lowpass filter responses.
    """
    from filters.iir_filter_butterworth_lowpass import plot_lowpass_filter_responses
    plot_lowpass_filter_responses(order=4, cutoff=100, fs=1000)
    

def test_plot_lowpass_iir_opt_filter_responses():
    """
    Test the plotting of lowpass filter responses.
    """
    from filters.iir_filter_butterworth_lowpass import plot_lowpass_filter_opt_responses
    plot_lowpass_filter_opt_responses(order = 4, cutoff=100, fs=1000)

def test_plot_highpass_iir_opt_filter_responses():
    """
    Test the plotting of optimized highpass filter responses.
    """
    from filters.iir_filter_butterworth_highpass import plot_iir_highpass_filter_opt_responses
    plot_iir_highpass_filter_opt_responses(order = 4, cutoff=100, fs=1000)

def test_plot_highpass_iir_filter_responses():
    """
    Test the plotting of highpass filter responses.
    """
    from filters.iir_filter_butterworth_highpass import plot_frequency_response
    plot_frequency_response(order = 4, cutoff=100, fs=1000)


def test_plot_bandpass_iir_filter_opt_responses():
    """
    Test the plotting of bandpass filter responses.
    """
    from filters.iir_filter_butterworth_bandpass import plot_iir_bandpass_filter_opt_responses
    plot_iir_bandpass_filter_opt_responses(order=4, low_cutoff=100, high_cutoff=300, fs=1000)

def test_plot_lowpass_iir_filter_coefficients():
    """
    Test the plotting of lowpass filter coefficients.
    """
    from filters.iir_filter_butterworth_lowpass import plot_coefficients
    plot_coefficients(order = 4, cutoff=100, fs = 1000)

def test_plot_lowpass_filter_opt_coefficients():
    """
    Test the plotting of lowpass filter coefficients.
    """
    from filters.iir_filter_butterworth_lowpass import plot_opt_coefficients
    plot_opt_coefficients(order = 4, cutoff=100, fs=1000)

def test_plot_highpass_iir_filter_coefficients():
    """
    Test the plotting of highpass filter coefficients.
    """
    from filters.iir_filter_butterworth_highpass import plot_coefficients
    plot_coefficients(order = 4, cutoff=100, fs = 1000)


def test_plot_highpass_iir_filter_opt_coefficients():
    """
    Test the plotting of optimized highpass filter coefficients.
    """
    from filters.iir_filter_butterworth_highpass import plot_iir_highpass_filter_opt_coefficients
    plot_iir_highpass_filter_opt_coefficients(order = 4, cutoff = 100, fs = 1000)

def test_plot_bandpass_iir_opt_filter_coefficients():
    """
    Test the plotting of bandpass filter coefficients.
    """
    from filters.iir_filter_butterworth_bandpass import plot_iir_bandpass_filter_opt_coefficients
    plot_iir_bandpass_filter_opt_coefficients(order=4, low_cutoff=100, high_cutoff=300, fs=1000)

def test_plot_bandpass_iir_filter_coefficients():
    """
    Test the plotting of optimized IIR bandpass filter coefficients.
    """
    from filters.iir_filter_butterworth_bandpass import plot_bandpass_iir_filter_coefficients
    plot_bandpass_iir_filter_coefficients(order=4, low_cutoff=100, high_cutoff=300, fs=1000)

    import pytest

def test_validate_inputs_zero_fs():
    """Testiraj da li `validate_inputs()` diže iznimku kad je fs 0"""
    with pytest.raises(FilterErrorBp, match="Sampling frequency (fs) cannot be zero."):
        validate_inputs(4, 50, 150, 0)

def test_validate_inputs_zero_cutoff():
    """Testiraj da li `validate_inputs()` diže iznimku kad je jedan od cutoff-a 0"""
    with pytest.raises(FilterErrorBp, match="Cutoff frequencies (lowcut, highcut) cannot be zero."):
        validate_inputs(4, 0, 150, 1000)

def test_validate_inputs_invalid_order():
    """Testiraj da li `validate_inputs()` diže iznimku za nevalidan order (nepozitivan)"""
    with pytest.raises(FilterErrorBp, match="Order must be a positive integer."):
        validate_inputs(-4, 50, 150, 1000)

def test_validate_inputs_invalid_lowcut():
    """Testiraj da li `validate_inputs()` diže iznimku za nevalidnu lowcut vrijednost"""
    with pytest.raises(FilterErrorBp, match="Lowcut frequency must be a positive number."):
        validate_inputs(4, -50, 150, 1000)

def test_validate_inputs_invalid_highcut():
    """Testiraj da li `validate_inputs()` diže iznimku za nevalidnu highcut vrijednost"""
    with pytest.raises(FilterErrorBp, match="Highcut frequency must be a positive number."):
        validate_inputs(4, 50, -150, 1000)

def test_validate_inputs_highcut_greater_than_nyquist():
    """Testiraj da li `validate_inputs()` diže iznimku ako je highcut >= fs/2"""
    with pytest.raises(FilterErrorBp, match="Highcut frequency must be less than half the sampling frequency."):
        validate_inputs(4, 50, 5000, 10000)

def test_validate_inputs_lowcut_greater_than_nyquist():
    """Testiraj da li `validate_inputs()` diže iznimku ako je lowcut >= fs/2"""
    with pytest.raises(FilterErrorBp, match="Lowcut frequency must be less than half the sampling frequency."):
        validate_inputs(4, 5000, 15000, 10000)

def test_validate_inputs_lowcut_ge_highcut():
    """Testiraj da li `validate_inputs()` diže iznimku ako je lowcut >= highcut"""
    with pytest.raises(FilterErrorBp, match="Lowcut frequency must be less than highcut frequency."):
        validate_inputs(4, 150, 100, 1000)

def test_butterworth_bp_manual_normal():
    """Test za generisanje Butterworth filtera sa ručnom metodom i normalnim parametrima"""
    b, a = butterworth_bp_manual(4, 50, 150, 1000)
    assert len(b) == 5
    assert len(a) == 5
    assert np.allclose(b, [expected_b_coeffs])  # zamijeni sa stvarnim vrijednostima

def test_butterworth_bp_builtin_comparison():
    """Test za upoređivanje `butterworth_bp_builtin` sa `butterworth_bp_manual` metodom"""
    b_manual, a_manual = butterworth_bp_manual(4, 50, 150, 1000)
    b_builtin, a_builtin = butterworth_bp_builtin(4, 50, 150, 1000)
    assert np.allclose(b_manual, b_builtin)
    assert np.allclose(a_manual, a_builtin)

def test_butterworth_bp_manual_opt():
    """Test za optimizovanu verziju Butterworth filtera"""
    b, a = butterworth_bp_manual_opt(4, 50, 150, 1000)
    assert len(b) == 5
    assert len(a) == 5
    assert np.allclose(b, [expected_b_opt])  # zamijeni sa stvarnim vrijednostima

def test_plot_bandpass_filter_response():
    """Test za plotanje odgovora filtra"""
    plot_iir_bandpass_filter_opt_responses(4, 50, 150, 1000)  # Osiguraj da plotanje ne izaziva greške

def test_plot_bandpass_filter_coefficients():
    """Test za plotanje koeficijenata filtra"""
    plot_iir_bandpass_filter_opt_coefficients(4, 50, 150, 1000)  # Osiguraj da plotanje koeficijenata ne izaziva greške

def test_plot_bandpass_manual_coefficients():
    """Test za plotanje koeficijenata samo manualne metode"""
    plot_bandpass_iir_filter_coefficients(4, 50, 150, 1000)  # Osiguraj da plotanje koeficijenata ne izaziva greške

def test_poles_and_zeros_stability():
    """Test za provjeru stabilnosti polova i nula"""
    b, a = butterworth_bp_builtin(4, 50, 150, 1000)
    zeros, poles, gain = tf2zpk(b, a)
    assert np.all(np.abs(poles) < 1)  # Provjerava da li su polovi unutar jedinicne kružnice

def test_filter_gain_check():
    """Test za provjeru gain-a filtra"""
    b, a = butterworth_bp_builtin(4, 50, 150, 1000)
    zeros, poles, gain = tf2zpk(b, a)
    assert np.isclose(gain, 1.0, atol=1e-3)  # Provjerava da li je gain blizu 1

def test_edge_case_filter():
    """Test za filtriranje sa vrlo malim ili velikim frekvencijama"""
    b, a = butterworth_bp_manual(4, 50, 150, 1e6)  # Testiranje sa vrlo visokim fs
    assert len(b) > 0
    assert len(a) > 0

def test_invalid_order_for_filter():
    """Test za nevalidne vrijednosti order-a (0 ili negativne vrijednosti)"""
    with pytest.raises(FilterErrorBp, match="Order must be a positive integer."):
        butterworth_bp_manual(0, 50, 150, 1000)

    with pytest.raises(FilterErrorBp, match="Order must be a positive integer."):
        butterworth_bp_manual(-4, 50, 150, 1000)
