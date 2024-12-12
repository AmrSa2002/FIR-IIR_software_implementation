import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter, firwin, freqz
from filters.fir_filter_lowpass import lowpass_fir_filter_opt_manual
from filters.fir_filter_bandpass import bandpass_fir_filter_opt_manual
from filters.fir_filter_highpass import highpass_fir_filter_opt_manual
from filters.iir_filter_butterworth_highpass import butterworth_hp_manual
from filters.iir_filter_butterworth_lowpass import butterworth_lp_manual

sampling_rate = 1000

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

def fft_analysis(signal, sampling_rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_magnitude = np.abs(np.fft.rfft(signal))
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

    t, signal = generate_signal(signal_type, frequency)
    filtered_signal, b, a = apply_filter(filter_type, signal, **filter_params)

    freq, signal_fft = fft_analysis(signal, sampling_rate)
    _, filtered_signal_fft = fft_analysis(filtered_signal, sampling_rate)

    w, h = freqz(b, a, worN=8000)
    freq_response = w * sampling_rate / (2 * np.pi)

    fig, axs = plt.subplots(3, 2, figsize=(10, 9))

    axs[0, 0].plot(t, signal)
    axs[0, 0].set_title('Original Signal (Time Domain)')
    axs[1, 0].plot(freq, signal_fft)
    axs[1, 0].set_title('Original Signal (Frequency Domain)')

    axs[0, 1].plot(t, filtered_signal)
    axs[0, 1].set_title('Filtered Signal (Time Domain)')
    axs[1, 1].plot(freq, filtered_signal_fft)
    axs[1, 1].set_title('Filtered Signal (Frequency Domain)')

    axs[2, 0].plot(freq_response, np.abs(h))
    axs[2, 0].set_title('Filter Frequency Response')
    axs[2, 0].set_xlabel('Frequency (Hz)')
    axs[2, 0].set_ylabel('Gain')

    plt.tight_layout()
    plt.show()

def show_frame(frame):
    frame.tkraise()

def on_signal_type_change(*args):
    if signal_type.get():
        next_button1.config(state='normal')
    else:
        next_button1.config(state='disabled')

def on_frequency_change(*args):
    if frequency_entry.get():
        next_button2.config(state='normal')
    else:
        next_button2.config(state='disabled')

def on_filter_type_change(*args):
    if filter_type.get():
        next_button3.config(state='normal')
    else:
        next_button3.config(state='disabled')

def on_cutoff_change(*args):
    if cutoff_entry.get():
        plot_button.config(state='normal')
    else:
        plot_button.config(state='disabled')

def on_num_taps_change(*args):
    if num_taps_entry.get():
        plot_button.config(state='normal')
    else:
        plot_button.config(state='disabled')

def on_lowcut_change(*args):
    if lowcut_entry.get():
        plot_button.config(state='normal')
    else:
        plot_button.config(state='disabled')

def on_highcut_change(*args):
    if highcut_entry.get():
        plot_button.config(state='normal')
    else:
        plot_button.config(state='disabled')

def on_order_change(*args):
    if order_entry.get():
        plot_button.config(state='normal')
    else:
        plot_button.config(state='disabled')

def on_sampling_rate_change(*args):
    if sampling_rate_entry.get():
        plot_button.config(state='normal')
    else:
        plot_button.config(state='disabled')

root = tk.Tk()
root.title("Filter GUI")

root.geometry("400x300")


signal_frame = ttk.Frame(root, padding="10 10 10 10")
frequency_frame = ttk.Frame(root, padding="10 10 10 10")
filter_frame = ttk.Frame(root, padding="10 10 10 10")
filter_params_frame = ttk.Frame(root, padding="10 10 10 10")

for frame in (signal_frame, frequency_frame, filter_frame, filter_params_frame):
    frame.grid(row=0, column=0, sticky='nsew')


signal_type_label = ttk.Label(signal_frame, text="Tip signala:")
signal_type_label.pack(pady=10)
signal_type = tk.StringVar()
signal_type.trace('w', on_signal_type_change)
signal_type_combobox = ttk.Combobox(signal_frame, textvariable=signal_type)
signal_type_combobox['values'] = ("Pravougaoni", "Sinusni", "Sinusni sa šumom", "Pravougaoni sa šumom", "Višefrekvencijski")
signal_type_combobox.pack()
next_button1 = ttk.Button(signal_frame, text="Next", command=lambda: show_frame(frequency_frame), state='disabled')
next_button1.pack(pady=20)


frequency_label = ttk.Label(frequency_frame, text="Frekvencija (Hz):")
frequency_label.pack(pady=10)
frequency_entry = ttk.Entry(frequency_frame)
frequency_entry.pack()
frequency_entry.bind('<KeyRelease>', on_frequency_change)
next_button2 = ttk.Button(frequency_frame, text="Next", command=lambda: show_frame(filter_frame), state='disabled')
next_button2.pack(pady=20)


filter_label = ttk.Label(filter_frame, text="Filter:")
filter_label.pack(pady=10)
filter_type = tk.StringVar()
filter_type.trace('w', on_filter_type_change)
filter_combobox = ttk.Combobox(filter_frame, textvariable=filter_type)
filter_combobox['values'] = ("Lowpass_FIR", "Bandpass_FIR", "Highpass_FIR", "Lowpass_IIR", "Highpass_IIR")
filter_combobox.pack()
next_button3 = ttk.Button(filter_frame, text="Next", command=lambda: show_frame(filter_params_frame), state='disabled')
next_button3.pack(pady=20)


def show_filter_params():
    for widget in filter_params_frame.winfo_children():
        widget.destroy()

    filter_type = filter_combobox.get()
    global plot_button, cutoff_entry, num_taps_entry, lowcut_entry, highcut_entry, order_entry, sampling_rate_entry

    if filter_type == "Lowpass_FIR" or filter_type == "Highpass_FIR":
        cutoff_label = ttk.Label(filter_params_frame, text="Cutoff frekvencija (Hz):")
        cutoff_label.pack(pady=10)
        cutoff_entry = ttk.Entry(filter_params_frame)
        cutoff_entry.pack()
        cutoff_entry.bind('<KeyRelease>', on_cutoff_change)

        num_taps_label = ttk.Label(filter_params_frame, text="Broj tapova:")
        num_taps_label.pack(pady=10)
        num_taps_entry = ttk.Entry(filter_params_frame)
        num_taps_entry.pack()
        num_taps_entry.bind('<KeyRelease>', on_num_taps_change)

    elif filter_type == "Bandpass_FIR":
        lowcut_label = ttk.Label(filter_params_frame, text="Lowcut frekvencija (Hz):")
        lowcut_label.pack(pady=10)
        lowcut_entry = ttk.Entry(filter_params_frame)
        lowcut_entry.pack()
        lowcut_entry.bind('<KeyRelease>', on_lowcut_change)

        highcut_label = ttk.Label(filter_params_frame, text="Highcut frekvencija (Hz):")
        highcut_label.pack(pady=10)
        highcut_entry = ttk.Entry(filter_params_frame)
        highcut_entry.pack()
        highcut_entry.bind('<KeyRelease>', on_highcut_change)

        num_taps_label = ttk.Label(filter_params_frame, text="Broj tapova:")
        num_taps_label.pack(pady=10)
        num_taps_entry = ttk.Entry(filter_params_frame)
        num_taps_entry.pack()
        num_taps_entry.bind('<KeyRelease>', on_num_taps_change)

    elif filter_type == "Lowpass_IIR" or filter_type == "Highpass_IIR":
        order_label = ttk.Label(filter_params_frame, text="Red filtera:")
        order_label.pack(pady=10)
        order_entry = ttk.Entry(filter_params_frame)
        order_entry.pack()
        order_entry.bind('<KeyRelease>', on_order_change)

        cutoff_label = ttk.Label(filter_params_frame, text="Cutoff frekvencija (Hz):")
        cutoff_label.pack(pady=10)
        cutoff_entry = ttk.Entry(filter_params_frame)
        cutoff_entry.pack()
        cutoff_entry.bind('<KeyRelease>', on_cutoff_change)

        sampling_rate_label = ttk.Label(filter_params_frame, text="Sampling rate (Hz):")
        sampling_rate_label.pack(pady=10)
        sampling_rate_entry = ttk.Entry(filter_params_frame)
        sampling_rate_entry.pack()
        sampling_rate_entry.bind('<KeyRelease>', on_sampling_rate_change)

    plot_button = ttk.Button(filter_params_frame, text="Prikaži signal", command=plot_signals, state='disabled')
    plot_button.pack(pady=20)

next_button3.config(command=lambda: [show_frame(filter_params_frame), show_filter_params()])


show_frame(signal_frame)

root.mainloop()
