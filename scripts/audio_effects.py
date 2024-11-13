import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import lfilter
import os
from scipy.io import wavfile
import soundfile as sf  # Required for saving in multiple formats including .wav and .flac
from pydub import AudioSegment  # Required for saving .mp3, .aac, and other formats

def apply_audio_effects(audio_signal, effect_reverb=True, effect_echo=False, effect_eq=False, effect_flanger=False, effect_chorus=False,
                        plot_freq=False, plot_time=False, save_path=None, save_audio_file_as=None, sample_rate=44100):
    """
    Apply selected audio effects (reverb, echo, EQ, flanger, chorus) to an audio signal and save in the specified format.

    Parameters:
    audio_signal (numpy array): Input audio signal.
    effect_reverb (bool): If True, apply reverb effect. Default is True.
    effect_echo (bool): If True, apply echo effect. Default is False.
    effect_eq (bool): If True, apply EQ (low-pass filter) effect. Default is False.
    effect_flanger (bool): If True, apply flanger effect. Default is False.
    effect_chorus (bool): If True, apply chorus effect. Default is False.
    plot_freq (bool): If True, plots the audio signal in the frequency domain. Default is False.
    plot_time (bool): If True, plots the audio signal in the time domain. Default is False.
    save_path (str): Path to save the processed audio signal and plots. If None, saves in the current directory.
    save_audio_file_as (str): Specify format to save audio file (e.g., "wav", "mp3", "aac", "flac"). If None, audio is not saved.
    sample_rate (int): The sample rate of the signals (default is 44100 Hz).

    Returns:
    processed_signal (numpy array): Result of the audio effect.
    final_save_path (str): The path where the processed audio is saved.
    """
    processed_signal = audio_signal.copy()

    if effect_reverb:
        delay_samples = int(0.03 * sample_rate)  # 30ms delay
        decay_factor = 0.5
        for i in range(delay_samples, len(audio_signal)):
            processed_signal[i] += decay_factor * audio_signal[i - delay_samples]

    if effect_echo:
        delay_samples = int(0.5 * sample_rate)  # 500ms delay
        decay_factor = 0.4
        for i in range(delay_samples, len(audio_signal)):
            processed_signal[i] += decay_factor * audio_signal[i - delay_samples]

    if effect_eq:
        cutoff = 5000  # Cutoff frequency in Hz
        normalized_cutoff = cutoff / (0.5 * sample_rate)
        b = [normalized_cutoff]  # Low-pass filter coefficients
        a = [1, normalized_cutoff - 1]
        processed_signal = lfilter(b, a, processed_signal)

    if effect_flanger:
        delay_samples = int(0.005 * sample_rate)  # 5ms delay
        modulation_frequency = 0.25  # Modulation frequency in Hz
        modulated_signal = np.zeros_like(audio_signal)
        for i in range(len(audio_signal)):
            modulated_delay = int(delay_samples * (1 + np.sin(2 * np.pi * modulation_frequency * i / sample_rate)))
            if i >= modulated_delay:
                modulated_signal[i] = audio_signal[i - modulated_delay]
        processed_signal += modulated_signal * 0.7

    if effect_chorus:
        delay_samples = int(0.02 * sample_rate)  # 20ms delay
        modulation_frequency = 0.1
        modulated_signal = np.zeros_like(audio_signal)
        for i in range(len(audio_signal)):
            modulated_delay = int(delay_samples * (1 + np.sin(2 * np.pi * modulation_frequency * i / sample_rate)))
            if i >= modulated_delay:
                modulated_signal[i] = audio_signal[i - modulated_delay]
        processed_signal += modulated_signal * 0.5

    # Plot frequency domain if required
    if plot_freq:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.abs(fft(audio_signal)))
        plt.title("Original Audio Signal (Frequency Domain)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        
        plt.subplot(1, 2, 2)
        plt.plot(np.abs(fft(processed_signal)))
        plt.title("Processed Audio Signal (Frequency Domain)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        
        plt.tight_layout()
        plt.show()

    # Plot time domain if required
    if plot_time:
        time_audio = np.linspace(0, len(audio_signal) / sample_rate, len(audio_signal))
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(time_audio, audio_signal)
        plt.title("Original Audio Signal (Time Domain)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.subplot(1, 2, 2)
        plt.plot(time_audio, processed_signal)
        plt.title("Processed Audio Signal (Time Domain)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

    # Determine save path
    if save_path is None:
        save_path = os.getcwd()
    final_save_path = os.path.join(save_path, "processed_audio_effect")

    # Save audio file in the specified format
    if save_audio_file_as:
        final_save_path += f".{save_audio_file_as.lower()}"
        if save_audio_file_as.lower() == "wav":
            wavfile.write(final_save_path, sample_rate, processed_signal.astype(np.int16))
        elif save_audio_file_as.lower() == "mp3":
            audio_segment = AudioSegment(
                processed_signal.tobytes(), frame_rate=sample_rate, sample_width=processed_signal.dtype.itemsize, channels=1
            )
            audio_segment.export(final_save_path, format="mp3")
        elif save_audio_file_as.lower() == "aac":
            audio_segment = AudioSegment(
                processed_signal.tobytes(), frame_rate=sample_rate, sample_width=processed_signal.dtype.itemsize, channels=1
            )
            audio_segment.export(final_save_path, format="aac")
        else:
            sf.write(final_save_path, processed_signal, sample_rate, format=save_audio_file_as.lower())

    return processed_signal, final_save_path

# Example usage:
sample_rate = 44100  # Sampling rate in Hz
duration = 2.0  # Duration in seconds
audio_signal = np.random.randn(int(sample_rate * duration))  # Random audio signal for testing

# Apply effects and save as .aac file
processed_signal, save_path = apply_audio_effects(audio_signal, effect_reverb=True, effect_echo=True, save_audio_file_as="aac", plot_freq=True, plot_time=True)
print("Processed signal saved at:", save_path)
