import timeit
import numpy as np
import matplotlib.pyplot as plt
from filters.fir_filter_highpass import highpass_fir_filter_manual
from filters.fir_filter_highpass_opt import highpass_fir_filter_opt_manual
from filters.fir_filter_lowpass import lowpass_fir_filter_manual
from filters.fir_filter_lowpass_opt import lowpass_fir_filter_opt_manual
from filters.fir_filter_bandpass import bandpass_fir_filter_manual
from filters.fir_filter_bandpass_opt import bandpass_fir_filter_opt_manual

def measure_performance_highpass():
    """
    Measures the execution time of the original and optimized high-pass FIR filter design functions
    to evaluate the performance improvement.
    """
    cutoff_freq = 0.51  # Normalized cutoff frequency for the filter
    num_taps = 101  # Number of taps for a more significant difference in measurement

    # Measure execution time of the original function
    original_time = timeit.timeit(
        stmt="highpass_fir_filter_manual(cutoff_freq, num_taps)",
        setup=(
            "from filters.fir_filter_highpass import highpass_fir_filter_manual;"
            "cutoff_freq = 0.51; num_taps = 101"
        ),
        number=1000
    )

    # Measure execution time of the optimized function
    optimized_time = timeit.timeit(
        stmt="highpass_fir_filter_opt_manual(cutoff_freq, num_taps)",
        setup=(
            "from filters.fir_filter_highpass_opt import highpass_fir_filter_opt_manual;"
            "cutoff_freq = 0.51; num_taps = 101"
        ),
        number=1000
    )

    # Print the results
    print(f"HIGHPASS FIR - Original function time: {original_time:.6f} seconds")
    print(f"HIGHPASS FIR - Optimized function time: {optimized_time:.6f} seconds")
    print(f"HIGHPASS FIR - Performance improvement: {((original_time - optimized_time) / original_time) * 100:.2f}%")

def measure_performance_lowpass():
    """
    Measures the execution time of the original and optimized high-pass FIR filter design functions
    to evaluate the performance improvement.
    """
    cutoff_freq = 0.51  # Normalized cutoff frequency for the filter
    num_taps = 101  # Number of taps for a more significant difference in measurement

    # Measure execution time of the original function
    original_time = timeit.timeit(
        stmt="lowpass_fir_filter_manual(cutoff_freq, num_taps)",
        setup=(
            "from filters.fir_filter_lowpass import lowpass_fir_filter_manual;"
            "cutoff_freq = 0.51; num_taps = 101"
        ),
        number=1000
    )

    # Measure execution time of the optimized function
    optimized_time = timeit.timeit(
        stmt="lowpass_fir_filter_opt_manual(cutoff_freq, num_taps)",
        setup=(
            "from filters.fir_filter_lowpass_opt import lowpass_fir_filter_opt_manual;"
            "cutoff_freq = 0.51; num_taps = 101"
        ),
        number=1000
    )

    # Print the results
    print(f"LOWPASS FIR - Original function time: {original_time:.6f} seconds")
    print(f"LOWPASS FIR - Optimized function time: {optimized_time:.6f} seconds")
    print(f"LOWPASS FIR - Performance improvement: {((original_time - optimized_time) / original_time) * 100:.2f}%")

def measure_performance_bandpass():
    """
    Measures the execution time of the original and optimized high-pass FIR filter design functions
    to evaluate the performance improvement.
    """

    lowcut_bandpass = 0.2  # Lower cutoff frequency
    highcut_bandpass = 0.4  # Upper cutoff frequency
    num_taps = 101  # Number of taps for a more significant difference in measurement

    # Measure execution time of the original function
    original_time = timeit.timeit(
        stmt="bandpass_fir_filter_manual(lowcut_bandpass, highcut_bandpass, num_taps)",
        setup=(
            "from filters.fir_filter_bandpass import bandpass_fir_filter_manual;"
            "lowcut_bandpass = 0.2; highcut_bandpass = 0.4; num_taps = 101"
        ),
        number=1000
    )

    # Measure execution time of the optimized function
    optimized_time = timeit.timeit(
        stmt="bandpass_fir_filter_opt_manual(lowcut_bandpass, highcut_bandpass, num_taps)",
        setup=(
            "from filters.fir_filter_bandpass_opt import bandpass_fir_filter_opt_manual;"
            "lowcut_bandpass = 0.2; highcut_bandpass = 0.4; num_taps = 101"
        ),
        number=1000
    )

    # Print the results
    print(f"BANDPASS FIR - Original function time: {original_time:.6f} seconds")
    print(f"BANDPASS FIR - Optimized function time: {optimized_time:.6f} seconds")
    print(f"BANDPASS FIR - Performance improvement: {((original_time - optimized_time) / original_time) * 100:.2f}%")

# Call the performance measurement function
measure_performance_highpass()
measure_performance_lowpass()
measure_performance_bandpass()