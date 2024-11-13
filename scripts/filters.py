import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import os

def butter_filter(audio_signal, filter_type, cutoff_freqs, fs=44100, order=4):
    """
    Applies a Butterworth filter to an audio signal.

    Parameters:
    audio_signal (numpy array): Input audio signal.
    filter_type (str): Type of filter ('low', 'high', 'bandpass', 'allpass').
    cutoff_freqs (tuple or float): Cutoff frequency for LPF/HPF or (low, high) for BPF.
    fs (int): Sampling rate in Hz. Default is 44100.
    order (int): Order of the filter. Default is 4.

    Returns:
    filtered_signal (numpy array): The filtered audio signal.
    """
    nyquist = 0.5 * fs
    if filter_type == 'low':
        normal_cutoff = cutoff_freqs / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'high':
        normal_cutoff = cutoff_freqs / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == 'bandpass':
        low, high = cutoff_freqs
        normal_cutoff = [low / nyquist, high / nyquist]
        b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    elif filter_type == 'allpass':
        # All-pass filter (just phase shift, no amplitude change)
        b, a = butter(order, 0.1, btype='allpass', analog=False)
    else:
        raise ValueError("Unknown filter type. Please use 'low', 'high', 'bandpass', or 'allpass'.")
    
    # Apply the filter to the signal
    filtered_signal = lfilter(b, a, audio_signal)
    return filtered_signal


def apply_filters(audio_signal, apply_lpf=False, apply_hpf=False, apply_bpf=False, apply_apf=False, cutoff_freqs=None, plot_freq=False, plot_time=False, save_path=None, sample_rate=44100):
    """
    Apply LPF, HPF, BPF, and/or APF to an audio signal and save the results as .npy files.

    Parameters:
    audio_signal (numpy array): Input audio signal.
    apply_lpf (bool): If True, apply Low Pass Filter. Default is False.
    apply_hpf (bool): If True, apply High Pass Filter. Default is False.
    apply_bpf (bool): If True, apply Band Pass Filter. Default is False.
    apply_apf (bool): If True, apply All Pass Filter. Default is False.
    cutoff_freqs (tuple or float): Cutoff frequencies for the filters.
    plot_freq (bool): If True, plots the frequency domain of the filtered signal. Default is False.
    plot_time (bool): If True, plots the time domain of the filtered signal. Default is False.
    save_path (str): Path to save the filtered signal as .npy file. Default is current directory.
    sample_rate (int): The sample rate of the signals. Default is 44100 Hz.

    Returns:
    filtered_signal (numpy array): The filtered audio signal after applying the selected filters.
    final_save_path (str): The path where the processed audio is saved.
    """
    
    # Set default cutoff frequencies if not provided
    if cutoff_freqs is None:
        cutoff_freqs = {
            'lpf': 3000,  # Low-pass filter cutoff (3 kHz)
            'hpf': 300,   # High-pass filter cutoff (300 Hz)
            'bpf': (300, 3000),  # Band-pass filter cutoff (300 Hz - 3 kHz)
            'apf': 0.1    # All-pass filter cutoff (arbitrary, just phase shift)
        }
    
    filtered_signal = audio_signal.copy()
    
    # Apply Low Pass Filter if selected
    if apply_lpf:
        filtered_signal = butter_filter(filtered_signal, 'low', cutoff_freqs['lpf'], fs=sample_rate)
    
    # Apply High Pass Filter if selected
    if apply_hpf:
        filtered_signal = butter_filter(filtered_signal, 'high', cutoff_freqs['hpf'], fs=sample_rate)
    
    # Apply Band Pass Filter if selected
    if apply_bpf:
        filtered_signal = butter_filter(filtered_signal, 'bandpass', cutoff_freqs['bpf'], fs=sample_rate)
    
    # Apply All Pass Filter if selected
    if apply_apf:
        filtered_signal = butter_filter(filtered_signal, 'allpass', cutoff_freqs['apf'], fs=sample_rate)
    
    # Plot frequency domain if required
    if plot_freq:
        plt.figure(figsize=(15, 5))
        plt.plot(np.abs(np.fft.fft(filtered_signal)))
        plt.title("Filtered Signal (Frequency Domain)")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.show()
    
    # Plot time domain if required
    if plot_time:
        time_audio = np.linspace(0, len(filtered_signal) / sample_rate, len(filtered_signal))
        plt.figure(figsize=(15, 5))
        plt.plot(time_audio, filtered_signal)
        plt.title("Filtered Signal (Time Domain)")
        plt.xlabel("Time [sec]")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    # Determine save path
    if save_path is None:
        save_path = os.getcwd()
    
    final_save_path = os.path.join(save_path, "filtered_audio")
    
    # Save the filtered signal as .npy file
    np.save(final_save_path + "_filtered.npy", filtered_signal)
    
    return filtered_signal, final_save_path

# Example usage:
sample_rate = 44100  # Sampling rate in Hz
duration = 2.0  # Duration in seconds
audio_signal = np.random.randn(int(sample_rate * duration))  # Random audio signal for testing

# Apply LPF (Low Pass Filter) and save the result as .npy
filtered_signal, save_path = apply_filters(audio_signal, apply_lpf=True, apply_hpf=False, apply_bpf=False, apply_apf=False, save_path=".", plot_freq=True, plot_time=True)
print("Filtered signal saved at:", save_path + "_filtered.npy")
