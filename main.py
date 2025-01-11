import numpy as np
import matplotlib.pyplot as plt
from filters.fir_filter_lowpass import (
    plot_filter_responses,
    plot_filter_coefficients,
    plot_filter_opt_coefficients,
    plot_filter_opt_responses
)
from filters.fir_filter_bandpass import (
    plot_bandpass_filter_responses,
    plot_bandpass_filter_coefficients,
    plot_bandpass_filter_opt_coefficients,
    plot_bandpass_filter_opt_responses
)
from filters.fir_filter_highpass import (
    
    plot_highpass_filter_responses,
    plot_highpass_filter_coefficients,
    plot_highpass_filter_opt_coefficients,
    plot_highpass_filter_opt_responses,
    highpass_fir_filter_manual,
    highpass_fir_filter_opt_manual
)
from filters.iir_filter_butterworth_lowpass import  (
    plot_lowpass_filter_responses,
    plot_coefficients,
    plot_lowpass_filter_opt_responses,
    plot_opt_coefficients
)
from filters.iir_filter_butterworth_bandpass import (
    plot_bandpass_iir_filter_coefficients,
    plot_iir_bandpass_filter_opt_responses,
    plot_iir_bandpass_filter_opt_coefficients
)
from filters.iir_filter_butterworth_highpass import (
    plot_frequency_response,
    plot_coefficients,
    plot_iir_highpass_filter_opt_coefficients
)

# Parametri filtera
num_taps = 51
cutoff_freq_lowpass = 0.25
lowcut_bandpass = 0.2
highcut_bandpass = 0.4
cutoff_freq_highpass = 0.25
sample_rate = 1000
order = 4  # Order for IIR filters
cutoff_freq_highpass_iir=100
cutoff_freq_lowpass_iir=100
lowcut_bandpass_iir=100
highcut_bandpass_iir=300

def main():
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
    plot_bandpass_filter_coefficients(lowcut_bandpass, highcut_bandpass, num_taps)
    plot_bandpass_filter_responses(lowcut_bandpass, highcut_bandpass, num_taps, sample_rate)

    # Bandpass filter - optimized
    print("Bandpass Filter - Optimized")
    plot_bandpass_filter_opt_coefficients(lowcut_bandpass, highcut_bandpass, num_taps)
    plot_bandpass_filter_opt_responses(lowcut_bandpass, highcut_bandpass, num_taps, sample_rate)

    # Highpass filter
    print("Highpass Filter")
    plot_highpass_filter_coefficients(cutoff_freq_highpass, num_taps)
    plot_highpass_filter_responses(cutoff_freq_highpass, num_taps, sample_rate)

    # Highpass filter - optimized
    print("Highpass Filter - Optimized")
    plot_highpass_filter_opt_coefficients(cutoff_freq_highpass, num_taps)
    plot_highpass_filter_opt_responses(cutoff_freq_highpass, num_taps, sample_rate)

    # IIR Lowpass filter
    print("IIR Butterworth lowpass filter")
    plot_lowpass_filter_responses(order, cutoff_freq_lowpass_iir, sample_rate)
    plot_coefficients(order, cutoff_freq_lowpass_iir, sample_rate)

    # IIR Lowpass filter - optimized
    print("IIR Butterworth lowpass filter - Optimized")
    plot_lowpass_filter_opt_responses(order, cutoff_freq_lowpass_iir, sample_rate)
    plot_opt_coefficients(order, cutoff_freq_lowpass_iir, sample_rate)

     # IIR Highpass filter
    print("IIR Butterworth highpass filter")
    plot_frequency_response(order, cutoff_freq_highpass_iir, sample_rate)
    plot_coefficients(order, cutoff_freq_highpass_iir, sample_rate)
    
    # IIR Highpass filter - optimized
    print("IIR Butterworth highpass filter - Optimized")
    plot_frequency_response(order, cutoff_freq_highpass_iir, sample_rate)
    plot_iir_highpass_filter_opt_coefficients(order, cutoff_freq_highpass_iir, sample_rate)

     # IIR Bandpass filter
    print("IIR Butterworth bandpass filter")
    #plot_frequency_response(order, cutoff_freq_lowpass, cutoff_freq_highpass, sample_rate)
    plot_bandpass_iir_filter_coefficients(order, lowcut_bandpass_iir, highcut_bandpass_iir, sample_rate)
    

    # IIR Bandpass filter - optimized
    print("IIR Butterworth bandpass filter - Optimized")
    plot_iir_bandpass_filter_opt_responses(order, lowcut_bandpass_iir, highcut_bandpass_iir, sample_rate)
    plot_iir_bandpass_filter_opt_coefficients(order, lowcut_bandpass_iir, highcut_bandpass_iir, sample_rate)



   
    

if __name__ == "__main__":
    main()
