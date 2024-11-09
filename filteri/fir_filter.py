import numpy as np
from scipy import signal

def design_fir_filter(cutoff_freq=0.3, numtaps=51):
    """ Dizajnira FIR low-pass filter sa zadanim parametrima """
    fir_coeff = signal.firwin(numtaps, cutoff=cutoff_freq)
    return fir_coeff

def frequency_response(fir_coeff):
    """ Izračunava i vraća frekvencijski odziv """
    w, h = signal.freqz(fir_coeff)
    return w, h
