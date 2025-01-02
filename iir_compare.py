import timeit
import numpy as np
import matplotlib.pyplot as plt
from filters.iir_filter_butterworth_highpass import butterworth_hp_manual, butterworth_hp_manual_opt
from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual, butterworth_lp_manual_opt
from filters.iir_filter_butterworth_bandpass import butterworth_bp_manual, butterworth_bp_manual_opt
from memory_profiler import profile
import time
import psutil

# Function for measuring execution time for IIR high-pass filter
def measure_performance_highpass_iir():
    """
    Measure execution time for the original and optimized IIR high-pass filter functions.
    """
    order = 4  # Filter order
    cutoff = 0.51  # Normalized cutoff frequency
    fs = 1000  # Sampling frequency
    
    # Measuring time for the original function
    original_time = timeit.timeit(
        stmt="butterworth_hp_manual(order, cutoff, fs)",
        setup=( 
            "from filters.iir_filter_butterworth_highpass import butterworth_hp_manual;"
            "order = 4; cutoff = 0.51; fs = 1000"
        ),
        number=1000
    )
    
    # Measuring time for the optimized function
    optimized_time = timeit.timeit(
        stmt="butterworth_hp_manual_opt(order, cutoff, fs)",  # Assuming an optimized function exists
        setup=( 
            "from filters.iir_filter_butterworth_highpass import butterworth_hp_manual_opt;"
            "order = 4; cutoff = 0.51; fs = 1000"
        ),
        number=1000
    )
    

    # Print the results
    print(f"HIGHPASS IIR - Original function time: {original_time:.6f} seconds")
    print(f"HIGHPASS IIR - Optimized function time: {optimized_time:.6f} seconds")
    print(f"HIGHPASS IIR - Performance improvement: {((original_time - optimized_time) / original_time) * 100:.2f}%")


    # Function for measuring execution time for IIR low-pass filter
def measure_performance_lowpass_iir():
    """
    Measure execution time for the original and optimized IIR low-pass filter functions.
    """
    order = 4  # Filter order
    cutoff = 0.51  # Normalized cutoff frequency
    fs = 1000  # Sampling frequency
    
    # Measuring time for the original function
    original_time = timeit.timeit(
        stmt="butterworth_lp_manual(order, cutoff, fs)",
        setup=( 
            "from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual;"
            "order = 4; cutoff = 0.51; fs = 1000"
        ),
        number=1000
    )
    
    # Measuring time for the optimized function
    optimized_time = timeit.timeit(
        stmt="butterworth_lp_manual_opt(order, cutoff, fs)",
        setup=( 
            "from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual_opt;"
            "order = 4; cutoff = 0.51; fs = 1000"
        ),
        number=1000
    )

    # Print the results
    print(f"LOWPASS IIR - Original function time: {original_time:.6f} seconds")
    print(f"LOWPASS IIR - Optimized function time: {optimized_time:.6f} seconds")
    print(f"LOWPASS IIR - Performance improvement: {((original_time - optimized_time) / original_time) * 100:.2f}%")



def measure_performance_bandpass():
    """Measure execution time for original and optimized bandpass filters."""
    lowcut, highcut = 0.2, 0.51
    fs, order = 1000, 4

    original_time = timeit.timeit(
        stmt="butterworth_bp_manual(lowcut, highcut, fs, order)",
        setup="from filters.iir_filter_butterworth_bandpass import butterworth_bp_manual;"
              "lowcut, highcut, fs, order = 0.2, 0.51, 1000, 4",
        number=1000
    )

    optimized_time = timeit.timeit(
        stmt="butterworth_bp_manual_opt(lowcut, highcut, fs, order)",
        setup="from filters.iir_filter_butterworth_bandpass import butterworth_bp_manual_opt;"
              "lowcut, highcut, fs, order = 0.2, 0.51, 1000, 4",
        number=1000
    )

    print(f"BANDPASS IIR - Original function time: {original_time:.6f}s")
    print(f"BANDPASS IIR - Optimized function time: {optimized_time:.6f}s")
    print(f"BANDPASS IIR - Performance improvement: {((original_time - optimized_time) / original_time) * 100:.2f}%")

if __name__ == "__main__":
    measure_performance_highpass_iir()
    measure_performance_lowpass_iir() 
    measure_performance_bandpass()
    

# Function for profiling memory usage for IIR high-pass filter
@profile
def memory_iir_highpass():
    butterworth_hp_manual(4, 0.51, 1000)

@profile
def memory_iir_highpass_optimized():
    butterworth_hp_manual_opt(4, 0.51, 1000)

print("Memory usage for Butterworth Highpass IIR filter")
memory_iir_highpass()
print("\nMemory usage for Optimized Butterworth Highpass IIR filter")
memory_iir_highpass_optimized()

# Function for profiling memory usage for IIR low-pass filter
@profile
def memory_iir_lowpass():
    butterworth_lp_manual(4, 0.51, 1000)

@profile
def memory_iir_lowpass_optimized():
    butterworth_lp_manual_opt(4, 0.51, 1000)

print("\nMemory usage for Butterworth Lowpass IIR filter")
memory_iir_lowpass()
print("\nMemory usage for Optimized Butterworth Lowpass IIR filter")
memory_iir_lowpass_optimized()

@profile
def memory_bandpass():
    butterworth_bp_manual(0.2, 0.51, 1000, 4)

@profile
def memory_bandpass_opt():
    butterworth_bp_manual_opt(0.2, 0.51, 1000, 4)

# Memory usage comparison
print("\nMemory usage for Butterworth Bandpass IIR filter")
memory_bandpass()
print("\nMemory usage for Optimized Butterworth Bandpass IIR filter")
memory_bandpass_opt()

# Function for tracking memory usage
def track_memory_usage(func):
    """
    Track memory usage during function execution.
    """
    process = psutil.Process()  # Get the current process.
    memory_usage = []  # List to store memory usage
    elapsed_time = []  # List to store elapsed time

    def wrapper():
        start_time = time.time()  # Start time measurement
        # Periodically collect memory usage data before executing the function.
        for _ in range(50):
            memory_usage.append(process.memory_info().rss / 1024**2)  # Memory in MB
            elapsed_time.append(time.time() - start_time)
            time.sleep(0.01)  # Sampling interval
        # Execute the function
        func(4, 0.51, 1000)
        # Collect final memory and time data
        memory_usage.append(process.memory_info().rss / 1024**2)
        elapsed_time.append(time.time() - start_time)
        return memory_usage, elapsed_time

    return wrapper()

# Tracking memory for the original and optimized functions
memory_iir, time_iir = track_memory_usage(butterworth_hp_manual)
memory_iir_opt, time_iir_opt = track_memory_usage(butterworth_hp_manual_opt)
memory_iir_low, time_iir_low = track_memory_usage(butterworth_lp_manual)
memory_iir_low_opt, time_iir_low_opt = track_memory_usage(butterworth_lp_manual_opt)

# Plotting memory usage comparison for Highpass
plt.plot(time_iir, memory_iir, label="Highpass IIR Manual")
plt.plot(time_iir_opt, memory_iir_opt, label="Highpass IIR Optimized")
plt.xlabel("Time (s)")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Comparison for IIR Filters")
plt.legend()
plt.grid(True)
plt.show()

# Plotting memory usage comparison for Lowpass
plt.plot(time_iir_low, memory_iir_low, label="Lowpass IIR Manual")
plt.plot(time_iir_low_opt, memory_iir_low_opt, label="Lowpass IIR Optimized")
plt.xlabel("Time (s)")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Comparison for Lowpass IIR Filters")
plt.legend()
plt.grid(True)
plt.show()

def track_memory_usage_bp(func):
    """
    Track memory usage during function execution.
    """
    process = psutil.Process()  # Get the current process.
    memory_usage = []  # List to store memory usage
    elapsed_time = []  # List to store elapsed time

    def wrapper():
        start_time = time.time()  # Start time measurement
        # Periodically collect memory usage data before executing the function.
        for _ in range(50):
            memory_usage.append(process.memory_info().rss / 1024**2)  # Memory in MB
            elapsed_time.append(time.time() - start_time)
            time.sleep(0.01)  # Sampling interval
        # Execute the function
        func(0.2, 0.51, 1000, 4)
        # Collect final memory and time data
        memory_usage.append(process.memory_info().rss / 1024**2)
        elapsed_time.append(time.time() - start_time)
        return memory_usage, elapsed_time

    return wrapper()

lowcut, highcut, fs, order = 0.2, 0.51, 1000, 4
memory_bp, time_bp = track_memory_usage_bp(butterworth_bp_manual)
memory_bp_opt, time_bp_opt = track_memory_usage_bp(butterworth_bp_manual_opt)

# Plotting memory usage comparison for Bandpass

plt.plot(time_bp, memory_bp, label="Manual Bandpass")
plt.plot(time_bp_opt, memory_bp_opt, label="Optimized Bandpass")
plt.xlabel("Time (s)")
plt.ylabel("Memory (MB)")
plt.title("Bandpass Filter Memory Comparison")
plt.legend()
plt.grid()
#plt.show()




# Function for comparing filter coefficients
#def plot_filter_coefficients(cutoff_freq, order, fs):
  #  """
   # Compare coefficients of FIR and IIR filters.
   # """
   # b_iir, a_iir = butterworth_hp_manual(order, cutoff_freq, fs)
   # b_iir_opt, a_iir_opt = butterworth_hp_manual_opt(order, cutoff_freq, fs)

    #plt.figure(figsize=(12, 6))

    # Coefficients of the manual IIR filter
   # plt.subplot(2, 2, 1)
    #plt.stem(b_iir, label="b (numerator)")
    #plt.stem(a_iir, label="a (denominator)", markerfmt="C1o")
    #plt.title('Manual IIR Filter Coefficients')
    #plt.legend()

    # Coefficients of the optimized IIR filter
    #plt.subplot(2, 2, 2)
    #plt.stem(b_iir_opt, label="b (numerator)")
    #plt.stem(a_iir_opt, label="a (denominator)", markerfmt="C1o")
    #plt.title('Optimized IIR Filter Coefficients')
    #plt.legend()

    #plt.tight_layout()
   # plt.show()

# Calling the function to generate coefficient plots
#plot_filter_coefficients(0.51, 4, 1000)







# Function for comparing filter coefficients for Lowpass filters
def plot_filter_coefficients_lowpass(cutoff_freq, order, fs):
    """
    Compare coefficients of FIR and IIR filters for low-pass case.
    """
    b_iir, a_iir = butterworth_lp_manual(order, cutoff_freq, fs)
    b_iir_opt, a_iir_opt = butterworth_lp_manual_opt(order, cutoff_freq, fs)

    plt.figure(figsize=(12, 6))

    # Coefficients of the manual IIR filter
    plt.subplot(2, 2, 1)
    plt.stem(b_iir, label="b (numerator)")
    plt.stem(a_iir, label="a (denominator)", markerfmt="C1o")
    plt.title('Manual Lowpass IIR Filter Coefficients')
    plt.legend()

    # Coefficients of the optimized IIR filter
    plt.subplot(2, 2, 2)
    plt.stem(b_iir_opt, label="b (numerator)")
    plt.stem(a_iir_opt, label="a (denominator)", markerfmt="C1o")
    plt.title('Optimized Lowpass IIR Filter Coefficients')
    plt.legend()

    plt.tight_layout()
    plt.show()

















