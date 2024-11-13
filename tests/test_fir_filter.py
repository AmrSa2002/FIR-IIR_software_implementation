import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from filters.fir_filter import design_fir_filter
from filters.fir_filter_prvi import generate_fir_coefficients

num_taps=51
cutoff_freq=0.3
def test_design_fir_filter():
    fir_coeff = design_fir_filter(cutoff_freq, num_taps)
    assert len(fir_coeff) == 51, "Broj koeficijenata nije ispravan"


def test_fir_coefficients():
    fir_coefficients_manual = generate_fir_coefficients(num_taps, cutoff_freq)
    fir_coefficients_firwin = design_fir_filter(cutoff_freq, num_taps)
    assert fir_coefficients_manual == pytest.approx(fir_coefficients_firwin, rel=1e-7, abs=1e-3)
