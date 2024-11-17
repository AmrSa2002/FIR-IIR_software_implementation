import numpy as np
import matplotlib.pyplot as plt

from filters.fir_filter_highpass import design_highpass_fir_filter, manual_design_highpass_fir_filter


cutoff_freq = 0.3
numtaps = 21

# Dizajn filtera
fir_coeff = design_highpass_fir_filter(cutoff_freq, numtaps)
print((fir_coeff))


# Testiranje funkcija

fir_coeff2 = manual_design_highpass_fir_filter(cutoff_freq, numtaps)
print((fir_coeff2))

# Upoređivanje rezultata
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.stem(fir_coeff2, linefmt='b-', markerfmt='bo', basefmt='r-', label='Ručno generisani')
plt.title("Koeficijenti FIR filtera (Ručno)")
plt.xlabel("Index koeficijenta")
plt.ylabel("Vrijednost koeficijenta")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(fir_coeff, linefmt='g-', markerfmt='go', basefmt='r-', label='firwin')
plt.title("Koeficijenti FIR filtera (firwin)")
plt.xlabel("Index koeficijenta")
plt.ylabel("Vrijednost koeficijenta")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Crtanje frekvencijskog odziva
w, h_manual = np.linspace(0, 1, num=1000, endpoint=False), np.fft.fft(fir_coeff2, 1000)
w, h_firwin = np.linspace(0, 1, num=1000, endpoint=False), np.fft.fft(fir_coeff, 1000)
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
