import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter, freqz
from filters.fir_filter_lowpass import lowpass_fir_filter_opt_manual
from filters.fir_filter_bandpass import bandpass_fir_filter_opt_manual
from filters.fir_filter_highpass import highpass_fir_filter_opt_manual
from filters.iir_filter_butterworth_highpass import butterworth_hp_manual
from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual
from filters.iir_filter_butterworth_bandpass import butterworth_bp_manual_opt

sampling_rate = 1000  # Definisanje podrazumevane vrednosti za sampling_rate

def generate_signal(signal_type, frequency, duration=1.0, sampling_rate=1000):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    if signal_type == "Sinusni":
        return t, np.sin(2 * np.pi * frequency * t)
    elif signal_type == "Pravougaoni":
        return t, np.sign(np.sin(2 * np.pi * frequency * t))
    elif signal_type == "Sinusni sa šumom":
        return t, 0.5 * np.random.normal(0, 1, len(t)) + np.sin(2 * np.pi * frequency * t)
    elif signal_type == "Pravougaoni sa šumom":
        return t, 0.5 * np.random.normal(0, 1, len(t)) + np.sign(np.sin(2 * np.pi * frequency * t))
    elif signal_type == "Višefrekvencijski":
        return t, np.sin(2 * np.pi * frequency * t) + np.sin(2 * np.pi * frequency * t * 5) + np.sin(2 * np.pi * frequency * t * 10) + np.sin(2 * np.pi * frequency * t * 20)
    else:
        raise ValueError("Nepoznat tip signala!")

def apply_filter(filter_type, signal, cutoff_freq=0.5, num_taps=50, lowcut=0.2, highcut=0.4, order=4, sampling_rate=1000):
    nyquist = 0.5 * sampling_rate
    if filter_type == "Lowpass_FIR":
        fir_coeffs = lowpass_fir_filter_opt_manual(cutoff_freq / nyquist, num_taps)
        return lfilter(fir_coeffs, [1], signal), fir_coeffs, [1]
    elif filter_type == "Bandpass_FIR":
        fir_coeffs = bandpass_fir_filter_opt_manual(lowcut / nyquist, highcut / nyquist, num_taps)
        return lfilter(fir_coeffs, [1], signal), fir_coeffs, [1]
    elif filter_type == "Highpass_FIR":
        fir_coeffs = highpass_fir_filter_opt_manual(cutoff_freq / nyquist, num_taps)
        return lfilter(fir_coeffs, [1], signal), fir_coeffs, [1]
    elif filter_type == "Highpass_IIR":
        b, a = butterworth_hp_manual(order, cutoff_freq, sampling_rate)
        return lfilter(b, a, signal), b, a
    elif filter_type == "Lowpass_IIR":
        b, a = butterworth_lp_manual(order, cutoff_freq, sampling_rate)
        return lfilter(b, a, signal), b, a
    elif filter_type == "Bandass_IIR":
        b, a = butterworth_bp_manual_opt(lowcut, highcut, order, sampling_rate)
        return lfilter(b, a, signal), b, a

def fft_analysis(signal, sampling_rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/sampling_rate)  # Frekvencije (do Nyquistove)
    fft_magnitude = np.abs(np.fft.rfft(signal))  # Magnituda FFT-a
    return freq, fft_magnitude

def plot_signals():
    signal_type = signal_type_combobox.get()
    frequency = float(frequency_entry.get())
    filter_type = filter_combobox.get()
    filter_params = {}

    if filter_type == "Lowpass_FIR" or filter_type == "Highpass_FIR":
        filter_params['cutoff_freq'] = float(cutoff_entry.get())
        filter_params['num_taps'] = int(num_taps_entry.get())
    elif filter_type == "Bandpass_FIR":
        filter_params['lowcut'] = float(lowcut_entry.get())
        filter_params['highcut'] = float(highcut_entry.get())
        filter_params['num_taps'] = int(num_taps_entry.get())
    elif filter_type == "Lowpass_IIR" or filter_type == "Highpass_IIR":
        filter_params['order'] = int(order_entry.get())
        filter_params['cutoff_freq'] = float(cutoff_entry.get())
        filter_params['sampling_rate'] = float(sampling_rate_entry.get())
    elif filter_type == "Bandpass_IIR":
        filter_params['order'] = int(order_entry.get())
        filter_params['lowcut'] = float(lowcut_entry.get())
        filter_params['highcut'] = float(highcut_entry.get())
        filter_params['sampling_rate'] = float(sampling_rate_entry.get())

    t, signal = generate_signal(signal_type, frequency)
    filtered_signal, b, a = apply_filter(filter_type, signal, **filter_params)

    freq, signal_fft = fft_analysis(signal, sampling_rate)
    _, filtered_signal_fft = fft_analysis(filtered_signal, sampling_rate)

    w, h = freqz(b, a, worN=8000)
    freq_response = w * sampling_rate / (2 * np.pi)

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    axs[0, 0].plot(t, signal)
    axs[0, 0].set_title('Original Signal (Time Domain)', fontsize=14)
    axs[1, 0].plot(freq, signal_fft)
    axs[1, 0].set_title('Original Signal (Frequency Domain)', fontsize=14)

    axs[0, 1].plot(t, filtered_signal)
    axs[0, 1].set_title('Filtered Signal (Time Domain)', fontsize=14)
    axs[1, 1].plot(freq, filtered_signal_fft)
    axs[1, 1].set_title('Filtered Signal (Frequency Domain)', fontsize=14)

    axs[2, 0].plot(freq_response, np.abs(h))
    axs[2, 0].set_title('Filter Frequency Response', fontsize=14)
    axs[2, 0].set_xlabel('Frequency (Hz)', fontsize=12)
    axs[2, 0].set_ylabel('Gain', fontsize=12)

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, pady=20, sticky=tk.N+tk.S+tk.E+tk.W)

def on_filter_change(event):
    selected_filter = filter_combobox.get()
    if selected_filter == "Lowpass_FIR" or selected_filter == "Highpass_FIR":
        cutoff_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        cutoff_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        num_taps_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        num_taps_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        lowcut_label.grid_remove()
        lowcut_entry.grid_remove()
        highcut_label.grid_remove()
        highcut_entry.grid_remove()
        order_label.grid_remove()
        order_entry.grid_remove()
        sampling_rate_label.grid_remove()
        sampling_rate_entry.grid_remove()
    elif selected_filter == "Bandpass_FIR":
        lowcut_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        lowcut_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        highcut_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        highcut_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        num_taps_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        num_taps_entry.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        cutoff_label.grid_remove()
        cutoff_entry.grid_remove()
        order_label.grid_remove()
        order_entry.grid_remove()
        sampling_rate_label.grid_remove()
        sampling_rate_entry.grid_remove()
    elif selected_filter == "Lowpass_IIR" or selected_filter == "Highpass_IIR":
        order_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        order_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        cutoff_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        cutoff_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        sampling_rate_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        sampling_rate_entry.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        lowcut_label.grid_remove()
        lowcut_entry.grid_remove()
        highcut_label.grid_remove()
        highcut_entry.grid_remove()
        num_taps_label.grid_remove()
        num_taps_entry.grid_remove()
    elif selected_filter == "Bandpass_IIR":
        order_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        order_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        lowcut_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        lowcut_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        highcut_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        highcut_entry.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        num_taps_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        num_taps_entry.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
        cutoff_label.grid_remove()
        cutoff_entry.grid_remove()
        sampling_rate_label.grid_remove()
        sampling_rate_entry.grid_remove()

def fit_canvas_to_image(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
    canvas_width = max(event.width, 1300)  # Set a minimum width for the canvas
    canvas.itemconfig(canvas_window, width=canvas_width)

def on_mouse_wheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

root = tk.Tk()
root.title("Filter Design")

# Set the size of the window (width x height)
root.geometry("1250x768")  # Change the size as needed

# Create a style
style = ttk.Style()
style.configure("TFrame", background="#2596be")
style.configure("TLabel", background="#f5d197", font=("Arial", 12))
style.configure("TButton", background="#f5d197", font=("Arial", 12))
style.configure("TCombobox", font=("Arial", 12))

# Create a frame for the canvas and scrollbar
frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Add a canvas in that frame
canvas = tk.Canvas(frame, bg="#579c65")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the frame
scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.bind_all("<MouseWheel>", on_mouse_wheel)

# Create another frame inside the canvas
main_frame = ttk.Frame(canvas)
canvas_window = canvas.create_window((0, 0), window=main_frame, anchor="nw")

# Bind the configure event to fit the canvas to the image width
main_frame.bind("<Configure>", fit_canvas_to_image)

# Add your widgets to main_frame
signal_type_label = ttk.Label(main_frame, text="Tip signala:")
signal_type_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
signal_type = tk.StringVar()
signal_type_combobox = ttk.Combobox(main_frame, textvariable=signal_type)
signal_type_combobox['values'] = ("Pravougaoni", "Sinusni", "Sinusni sa šumom", "Pravougaoni sa šumom", "Višefrekvencijski")
signal_type_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

frequency_label = ttk.Label(main_frame, text="Frekvencija (Hz):")
frequency_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
frequency_entry = ttk.Entry(main_frame)
frequency_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

# Stranica za izbor filtera
filter_label = ttk.Label(main_frame, text="Filter:")
filter_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
filter_type = tk.StringVar()
filter_combobox = ttk.Combobox(main_frame, textvariable=filter_type)
filter_combobox['values'] = ("Lowpass_FIR", "Bandpass_FIR", "Highpass_FIR", "Lowpass_IIR", "Highpass_IIR", "Bandpass_IIR")
filter_combobox.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
filter_combobox.bind("<<ComboboxSelected>>", on_filter_change)

# Parametri filtera
cutoff_label = ttk.Label(main_frame, text="Cutoff frekvencija (Hz):")
cutoff_entry = ttk.Entry(main_frame)

num_taps_label = ttk.Label(main_frame, text="Broj tapova:")
num_taps_entry = ttk.Entry(main_frame)

lowcut_label = ttk.Label(main_frame, text="Lowcut frekvencija (Hz):")
lowcut_entry = ttk.Entry(main_frame)

highcut_label = ttk.Label(main_frame, text="Highcut frekvencija (Hz):")
highcut_entry = ttk.Entry(main_frame)

order_label = ttk.Label(main_frame, text="Red filtera:")
order_entry = ttk.Entry(main_frame)

sampling_rate_label = ttk.Label(main_frame, text="Sampling rate (Hz):")
sampling_rate_entry = ttk.Entry(main_frame)

plot_button = ttk.Button(main_frame, text="Prikaži signal", command=plot_signals)
plot_button.grid(row=7, column=0, columnspan=2, pady=20)

plot_frame = ttk.Frame(main_frame)
plot_frame.grid(row=8, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E, tk.N, tk.S))

root.mainloop()
