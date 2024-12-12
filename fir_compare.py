import timeit
import numpy as np
import matplotlib.pyplot as plt
from filters.fir_filter_highpass import highpass_fir_filter_manual, highpass_fir_filter_opt_manual
from filters.fir_filter_lowpass import lowpass_fir_filter_manual, lowpass_fir_filter_opt_manual
from filters.fir_filter_bandpass import bandpass_fir_filter_manual, bandpass_fir_filter_opt_manual
from memory_profiler import profile
import time
import psutil

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
            "from filters.fir_filter_highpass import highpass_fir_filter_opt_manual;"
            "cutoff_freq = 0.51; num_taps = 101"
        ),
        number=1000
    )
    print(f"Original function time: {original_time}")
    print(f"Optimized function time: {optimized_time}")

if __name__ == "__main__":
    measure_performance_highpass()

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
            "from filters.fir_filter_lowpass import lowpass_fir_filter_opt_manual;"
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
            "from filters.fir_filter_bandpass import bandpass_fir_filter_opt_manual;"
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
cut_off = 0.51
numtaps = 101
lowcut_bandpass = 0.2
highcut_bandpass = 0.4
@profile
def mem_lp():
    lowpass_fir_filter_manual(cut_off, numtaps)
@profile
def testiraj_optimizaciju():
    lowpass_fir_filter_opt_manual(cut_off, numtaps)
print("Memory usage for Lowpass FIR filter")
mem_lp()
print("\nMemory usage for Optimized Lowpass FIR filter")
testiraj_optimizaciju()
@profile
def memory_highpass_fir():
    highpass_fir_filter_manual(cut_off, numtaps)
@profile
def memory_highpass_fir_opt():
    highpass_fir_filter_opt_manual(cut_off, numtaps)
print("Memory usage for Highpass FIR filter")
memory_highpass_fir()
print("\nMemory usage for Optimized Highpass FIR filter")
memory_highpass_fir_opt()
@profile
def memory_bandpass_fir():
    bandpass_fir_filter_manual(lowcut_bandpass, highcut_bandpass, numtaps)
@profile
def memory_bandpass_fir_opt():
    bandpass_fir_filter_opt_manual(lowcut_bandpass, highcut_bandpass, numtaps)
print("Memory usage for Bandpass FIR filter")
memory_bandpass_fir()
print("\nMemory usage for Optimized Bandpass FIR filter")
memory_bandpass_fir_opt()
#Graphs
def track_memory_usage(func):
    """
    Tracks memory usage of the given function during its execution.
    Args:
        func (callable): The function whose memory usage is to be tracked.
        label (str): Label for the function, used for display purposes.
    Returns:
        tuple: A tuple containing two lists:
            - memory_usage (list): Memory usage in MB over time.
            - elapsed_time (list): Corresponding time intervals in seconds.
    """
    process = psutil.Process()  # Get the current process.
    memory_usage = []  # List to store memory usage data.
    elapsed_time = []  # List to store elapsed time data.
    def wrapper():
        start_time = time.time()  # Record the start time.
        # Collect memory data periodically before function execution.
        for _ in range(50):
            memory_usage.append(process.memory_info().rss / 1024**2)  # Memory in MB.
            elapsed_time.append(time.time() - start_time)
            time.sleep(0.01)  # Sampling interval.
        # Execute the function.
        func(cut_off, numtaps)
        # Collect final memory usage and elapsed time.
        memory_usage.append(process.memory_info().rss / 1024**2)
        elapsed_time.append(time.time() - start_time)
        return memory_usage, elapsed_time
    return wrapper()
memory_lowpass, time_loop = track_memory_usage(lowpass_fir_filter_manual)
memory_lowpass_opt, time_optimized = track_memory_usage(lowpass_fir_filter_opt_manual)
memory_highpass, time_loop = track_memory_usage(highpass_fir_filter_manual)
memory_highpass_opt, time_optimized = track_memory_usage(highpass_fir_filter_opt_manual)
# Plot memory usage comparison for Lowpass FIR
plt.plot(time_loop, memory_lowpass, label="Lowpass FIR")
plt.plot(time_optimized, memory_lowpass_opt, label="Optimized Lowpass FIR")
plt.xlabel("Time (s)")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Comparison")
plt.legend()
plt.grid(True)
plt.show()
# Plot memory usage comparison for Highpass FIR
plt.plot(time_loop, memory_highpass, label="Highpass FIR")
plt.plot(time_optimized, memory_highpass_opt, label="Optimized Highpass FIR")
plt.xlabel("Time (s)")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Comparison")
plt.legend()
plt.grid(True)
plt.show()
def track_memory_usage_bandpass(func):
    """
    Tracks memory usage of the given function during its execution.
    Args:
        func (callable): The function whose memory usage is to be tracked.
        label (str): Label for the function, used for display purposes.
    Returns:
        tuple: A tuple containing two lists:
            - memory_usage (list): Memory usage in MB over time.
            - elapsed_time (list): Corresponding time intervals in seconds.
    """
    process = psutil.Process()  # Get the current process.
    memory_usage = []  # List to store memory usage data.
    elapsed_time = []  # List to store elapsed time data.
    def wrapper():
        start_time = time.time()  # Record the start time.
        # Collect memory data periodically before function execution.
        for _ in range(50):
            memory_usage.append(process.memory_info().rss / 1024**2)  # Memory in MB.
            elapsed_time.append(time.time() - start_time)
            time.sleep(0.01)  # Sampling interval.
        # Execute the function.
        func(lowcut_bandpass, highcut_bandpass, numtaps)
        # Collect final memory usage and elapsed time.
        memory_usage.append(process.memory_info().rss / 1024**2)
        elapsed_time.append(time.time() - start_time)
        return memory_usage, elapsed_time
    return wrapper()
memory_bandpass, time_loop = track_memory_usage_bandpass(bandpass_fir_filter_manual)
memory_bandpass_opt, time_optimized = track_memory_usage_bandpass(bandpass_fir_filter_opt_manual)
# Plot memory usage comparison for Lowpass FIR
plt.plot(time_loop, memory_bandpass, label="Bandpass FIR")
plt.plot(time_optimized, memory_bandpass_opt, label="Optimized Bandpass FIR")
plt.xlabel("Time (s)")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Comparison")
plt.legend()
plt.grid(True)
plt.show()
