import pytest
from filteri.fir_filter import design_fir_filter, frequency_response

def test_design_fir_filter():
    fir_coeff = design_fir_filter()
    assert len(fir_coeff) == 51, "Broj koeficijenata nije ispravan"

def test_frequency_response():
    fir_coeff = design_fir_filter()
    w, h = frequency_response(fir_coeff)
    assert h[0] != 0, "Frekvencijski odziv nije ispravan"
