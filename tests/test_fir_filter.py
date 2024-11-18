import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from filters.fir_filter import design_fir_filter
from filters.fir_filter_prvi import generate_fir_coefficients
from filters.fir_filter_highpass import design_highpass_fir_filter, manual_design_highpass_fir_filter
from filters.fir_filter_bandpass import design_bandpass_fir_filter, manual_design_bandpass_fir_filter


def test_design_fir_filter():
    fir_coeff = design_fir_filter(cutoff_freq=0.3, numtaps=51)
    assert len(fir_coeff) == 51, "Broj koeficijenata nije ispravan"


def test_fir_coefficients():
    fir_coefficients_manual = generate_fir_coefficients(num_taps=51, cutoff_freq=0.3)
    fir_coefficients_firwin = design_fir_filter(cutoff_freq=0.3, numtaps=51)
    assert fir_coefficients_manual == pytest.approx(fir_coefficients_firwin, rel=1e-7, abs=1e-3)

def test_fir_coefficients_highpass():
    fir_coefficients_manual = design_highpass_fir_filter(cutoff_freq=0.3, numtaps=21)
    fir_coefficients_firwin = manual_design_highpass_fir_filter(cutoff_freq=0.3, numtaps=21)
    assert fir_coefficients_manual == pytest.approx(fir_coefficients_firwin, rel=1e-7, abs=1e-3)

def test_fir_coefficients_bandpass():
    fir_coefficients_manual = design_bandpass_fir_filter(low_cutoff=0.2, high_cutoff=0.4, numtaps=51)
    fir_coefficients_firwin = manual_design_bandpass_fir_filter(low_cutoff=0.2, high_cutoff=0.4, numtaps=51)
    assert fir_coefficients_manual == pytest.approx(fir_coefficients_firwin, rel=1e-7, abs=1e-3)
