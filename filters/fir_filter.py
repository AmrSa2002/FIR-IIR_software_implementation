import numpy as np
from scipy import signal

def design_fir_filter(cutoff_freq, numtaps):
    """ Dizajnira FIR low-pass filter sa zadanim parametrima """
    fir_coeff = signal.firwin(numtaps, cutoff=cutoff_freq)
    return fir_coeff
