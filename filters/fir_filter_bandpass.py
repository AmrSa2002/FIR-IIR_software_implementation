import numpy as np
from scipy.signal import firwin

# Funkcija za dizajn Bandpass FIR filtera koristeći firwin
def design_bandpass_fir_filter(low_cutoff, high_cutoff, numtaps):
    nyquist = 0.5  # Nyquist frekvencija (polovina uzorkovne frekvencije)
    return firwin(numtaps, [low_cutoff / nyquist, high_cutoff / nyquist], pass_zero=False)

# Ručna implementacija dizajniranja Bandpass FIR filtera (ovdje koristeći prozor metod)
def manual_design_bandpass_fir_filter(low_cutoff, high_cutoff, numtaps):
    nyquist = 0.5  # Nyquist frekvencija
    # Generisanje impulznih odgovora (idealni impulzni odgovor bandpass filtera)
    ideal_impulse_response = np.zeros(numtaps)
    mid_point = numtaps // 2
    for i in range(numtaps):
        # Idealni impulzni odgovor za bandpass filter
        if i == mid_point:
            ideal_impulse_response[i] = 2 * (high_cutoff - low_cutoff)
        elif i != mid_point:
            ideal_impulse_response[i] = (np.sin(2 * np.pi * high_cutoff * (i - mid_point)) -
                                         np.sin(2 * np.pi * low_cutoff * (i - mid_point))) / (np.pi * (i - mid_point))

    # Primjena prozora (npr. Hammingov prozor) na idealni impulzni odgovor
    window = np.hamming(numtaps)
    fir_coefficients = ideal_impulse_response * window

    return fir_coefficients
