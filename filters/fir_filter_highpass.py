import numpy as np
from scipy import signal

def design_highpass_fir_filter(cutoff_freq, numtaps):
    """ Dizajnira FIR high-pass filter sa zadanim parametrima """
    # Dizajniranje osnovnog low-pass filtera
    fir_coeff = signal.firwin(numtaps, cutoff=cutoff_freq, pass_zero=False)  # pass_zero=False za high-pass
    return fir_coeff


def manual_design_highpass_fir_filter(cutoff_freq, numtaps):
    """
    Dizajnira FIR high-pass filter sa zadanim parametrima, ručno.

    cutoff_freq: Normalizovana frekvencija odsecanja (0 do 1, gde je 1 Nyquistova frekvencija)
    numtaps: Broj koeficijenata (dužina filtera)
    """
    # Normalizovana ugaona frekvencija odsecanja

    wc = 1 * np.pi * cutoff_freq  # Normalizovana cutoff u radijanima

    # Srednji indeks (centar simetričnog filtra)
    M = numtaps // 2

    # Idealni impulsni odziv za high-pass filter
    h = np.zeros(numtaps)
    for n in range(numtaps):
        if n == M:  # Za srednji indeks
            h[n] = 1 - wc / np.pi
        else:
            h[n] = -np.sin(wc * (n - M)) / (np.pi * (n - M))
           # Primjena prozora (npr. Hammingov prozor)
        h[n] *= 0.54 - 0.46 * np.cos(2 * np.pi * n / (numtaps - 1))
    return h
