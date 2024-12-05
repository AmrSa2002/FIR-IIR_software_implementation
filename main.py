import numpy as np
import matplotlib.pyplot as plt
from filters.fir_filter_lowpass import lowpass_fir_filter_manual, lowpass_fir_filter_firwin, plot_filter_responses, plot_filter_coefficients
from filters.fir_filter_bandpass import bandpass_fir_filter_manual, bandpass_fir_filter_firwin, plot_bandpass_filter_responses, plot_bandpass_filter_coefficients
from filters.fir_filter_highpass import highpass_fir_filter_manual, highpass_fir_filter_firwin, plot_highpass_filter_responses, plot_highpass_filter_coefficients
from filters.fir_filter_highpass_opt import highpass_fir_filter_opt_manual, highpass_fir_filter_firwin, plot_highpass_filter_opt_responses, plot_highpass_filter_opt_coefficients
from filters.fir_filter_lowpass_opt import lowpass_fir_filter_opt_manual, lowpass_fir_filter_firwin, plot_filter_opt_coefficients, plot_filter_opt_responses
from filters.fir_filter_bandpass_opt import bandpass_fir_filter_opt_manual, bandpass_fir_filter_firwin, plot_bandpass_filter_opt_responses, plot_bandpass_filter_opt_coefficients


# Parametri filtera
num_taps = 51
cutoff_freq_lowpass = 0.25
lowcut_bandpass = 0.2
highcut_bandpass = 0.4
cutoff_freq_highpass = 0.25
sample_rate = 1000

# Lowpass filter
print("Lowpass Filter")
plot_filter_coefficients(cutoff_freq_lowpass, num_taps)
plot_filter_responses(cutoff_freq_lowpass, num_taps, sample_rate)

# Lowpass filter - optimized
print("Lowpass Filter - Optimized")
plot_filter_opt_coefficients(cutoff_freq_lowpass, num_taps)
plot_filter_opt_responses(cutoff_freq_lowpass, num_taps, sample_rate)

# Bandpass filter
print("Bandpass Filter")
plot_bandpass_filter_opt_coefficients(lowcut_bandpass, highcut_bandpass, num_taps)
plot_bandpass_filter_opt_responses(lowcut_bandpass, highcut_bandpass, num_taps, sample_rate)

# Highpass filter
print("Highpass Filter")
plot_highpass_filter_coefficients(cutoff_freq_highpass, num_taps)
plot_highpass_filter_responses(cutoff_freq_highpass, num_taps, sample_rate)

# Highpass filter optmizirani
print("Highpass Filter Optimized")
plot_highpass_filter_opt_coefficients(cutoff_freq_highpass, num_taps)
plot_highpass_filter_opt_responses(cutoff_freq_highpass, num_taps, sample_rate)