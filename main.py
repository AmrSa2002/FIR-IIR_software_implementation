import numpy as np
import matplotlib.pyplot as plt
from filteri.fir_filter import design_fir_filter, frequency_response

# Dizajn filtera
fir_coeff = design_fir_filter()

# Frekvencijski odziv
w, h = frequency_response(fir_coeff)

# Plotiranje
plt.plot(w/np.pi, 20 * np.log10(np.abs(h)))
plt.title('Frekvencijski odziv FIR filtera')
plt.xlabel('Normalizovana frekvencija [x/pi rad/sample]')
plt.ylabel('Magnituda [dB]')
plt.grid()
plt.show()
