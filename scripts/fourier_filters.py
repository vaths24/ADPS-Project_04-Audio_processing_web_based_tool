import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import stft
import os

def apply_transformations(audio_signal, apply_fft=True, apply_dft=False, apply_stft=False, plot_freq=False, plot_time=False, save_path=None, sample_rate=44100):
    """
    Apply FFT, DFT, and/or STFT transformations to an audio signal and save the results as .npy files.

    Parameters:
    audio_signal (numpy array): Input audio signal.
    apply_fft (bool): If True, apply FFT transformation. Default is True.
    apply_dft (bool): If True, apply DFT transformation. Default is False.
    apply_stft (bool): If True, apply STFT transformation. Default is False.
    plot_freq (bool): If True, plots the frequency domain of the transformed signal. Default is False.
    plot_time (bool): If True, plots the time domain of the transformed signal. Default is False.
    save_path (str): Path to save the transformed signal as .npy files. If None, saves in the current directory.
    sample_rate (int): The sample rate of the signals (default is 44100 Hz).

    Returns:
    transformed_signal (numpy array): The transformed audio signal after applying the selected transformations.
    final_save_path (str): The path where the processed audio is saved.
    """

    transformed_signal = audio_signal.copy()
    
    # Apply FFT if selected
    if apply_fft:
        transformed_signal = fft(audio_signal)

    # Apply DFT if selected
    if apply_dft:
        transformed_signal = np.fft.fft(audio_signal)

    # Apply STFT if selected
    if apply_stft:
        _, _, Zxx = stft(audio_signal, fs=sample_rate)
        transformed_signal = Zxx

    # Plot frequency domain if required
    if plot_freq:
        plt.figure(figsize=(15, 5))
        if apply_fft or apply_dft:
            plt.plot(np.abs(transformed_signal))
            plt.title("Transformed Signal (Frequency Domain)")
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
        elif apply_stft:
            plt.imshow(np.abs(transformed_signal), aspect="auto", cmap='jet', origin='lower', extent=[0, len(audio_signal) / sample_rate, 0, sample_rate / 2])
            plt.title("STFT of Signal (Frequency Domain)")
            plt.xlabel("Time [sec]")
            plt.ylabel("Frequency [Hz]")
        plt.tight_layout()
        plt.show()

    # Plot time domain if required
    if plot_time:
        time_audio = np.linspace(0, len(audio_signal) / sample_rate, len(audio_signal))
        plt.figure(figsize=(15, 5))
        plt.plot(time_audio, audio_signal)
        plt.title("Original Audio Signal (Time Domain)")
        plt.xlabel("Time [sec]")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    # Determine save path
    if save_path is None:
        save_path = os.getcwd()
    final_save_path = os.path.join(save_path, "fourier_transformed_audio")

    # Save the transformed signal as .npy file
    np.save(final_save_path + "_transformed.npy", transformed_signal)

    return transformed_signal, final_save_path

# Example usage:
sample_rate = 44100  # Sampling rate in Hz
duration = 2.0  # Duration in seconds
audio_signal = np.random.randn(int(sample_rate * duration))  # Random audio signal for testing

# Apply FFT and save the result as .npy
transformed_signal, save_path = apply_transformations(audio_signal, apply_fft=True, apply_dft=False, apply_stft=False, save_path=".", plot_freq=True)
print("Transformed signal saved at:", save_path + "_transformed.npy")
