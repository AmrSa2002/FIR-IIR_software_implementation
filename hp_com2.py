import timeit
import numpy as np
import matplotlib.pyplot as plt
from filters.fir_filter_highpass import highpass_fir_filter_manual
from filters.fir_filter_highpass_opt import highpass_fir_filter_opt_manual

def measure_performance():
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
    print(f"Original function time: {original_time:.6f} seconds")
    print(f"Optimized function time: {optimized_time:.6f} seconds")
    print(f"Performance improvement: {((original_time - optimized_time) / original_time) * 100:.2f}%")

# Call the performance measurement function
measure_performance()
