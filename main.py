import numpy as np
import matplotlib.pyplot as plt
from filters.fir_filter import design_fir_filter
from filters.fir_filter_prvi import generate_fir_coefficients

#Definicija parametara
num_taps = 51  # Broj koeficijenata (dužina filtera)
cutoff_freq = 0.2  # Normalizovana frekvencija odsecanja (0.0 do 1.0, gde je 1.0 Nyquist)
# Dizajn filtera

# Generisanje FIR koeficijenata
fir_coefficients_manual = generate_fir_coefficients(num_taps, cutoff_freq)

# Generisanje FIR koeficijenata koristeći firwin
fir_coefficients_firwin = design_fir_filter(cutoff_freq, num_taps)

# Upoređivanje rezultata
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.stem(fir_coefficients_manual, linefmt='b-', markerfmt='bo', basefmt='r-', label='Ručno generisani')
plt.title("Koeficijenti FIR filtera (Ručno)")
plt.xlabel("Index koeficijenta")
plt.ylabel("Vrijednost koeficijenta")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(fir_coefficients_firwin, linefmt='g-', markerfmt='go', basefmt='r-', label='firwin')
plt.title("Koeficijenti FIR filtera (firwin)")
plt.xlabel("Index koeficijenta")
plt.ylabel("Vrijednost koeficijenta")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Crtanje frekvencijskog odziva
w, h_manual = np.linspace(0, 1, num=1000, endpoint=False), np.fft.fft(fir_coefficients_manual, 1000)
w, h_firwin = np.linspace(0, 1, num=1000, endpoint=False), np.fft.fft(fir_coefficients_firwin, 1000)
h_manual = np.abs(h_manual[:500])
h_firwin = np.abs(h_firwin[:500])

plt.plot(w[:500], 20 * np.log10(h_manual + 1e-10), label="Ručno generisani FIR")
plt.plot(w[:500], 20 * np.log10(h_firwin + 1e-10), label="FIR generisan sa firwin")
plt.title("Frekvencijski odziv FIR Low Pass Filtera")
plt.xlabel("Normalizovana frekvencija (u odnosu na Nyquist)")
plt.ylabel("Magnituda (dB)")
plt.grid(True)
plt.legend()
plt.show()
