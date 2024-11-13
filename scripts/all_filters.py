import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter, stft
from scipy.io import wavfile
import os
import soundfile as sf  # For .wav and .flac saving
from pydub import AudioSegment  # For .mp3, .aac saving

def plot_signal(audio_signal, sample_rate, plot_freq=False, plot_time=False, transformed_signal=None, title=""):
    time_audio = np.linspace(0, len(audio_signal) / sample_rate, len(audio_signal))
    
    if plot_time:
        plt.figure(figsize=(10, 4))
        plt.plot(time_audio, audio_signal)
        plt.title(f"{title} (Time Domain)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()
        
    if plot_freq:
        freq_data = np.abs(fft(transformed_signal if transformed_signal is not None else audio_signal))
        plt.figure(figsize=(10, 4))
        plt.plot(freq_data)
        plt.title(f"{title} (Frequency Domain)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.show()

def apply_audio_effects(audio_signal, sample_rate=44100, effect_reverb=True, effect_echo=False, effect_eq=False, 
                        effect_flanger=False, effect_chorus=False, plot_freq=False, plot_time=False, 
                        save_path=None, save_audio_format=None):
    
    processed_signal = audio_signal.copy()

    # Apply effects with numpy operations for efficiency
    if effect_reverb:
        delay = int(0.03 * sample_rate)
        processed_signal[delay:] += 0.5 * audio_signal[:-delay]
    
    if effect_echo:
        delay = int(0.5 * sample_rate)
        processed_signal[delay:] += 0.4 * audio_signal[:-delay]
        
    if effect_eq:
        cutoff = 5000
        b, a = butter(1, cutoff / (0.5 * sample_rate), btype='low')
        processed_signal = lfilter(b, a, processed_signal)
    
    if effect_flanger or effect_chorus:
        mod_freq = 0.1 if effect_chorus else 0.25
        delay = int((0.02 if effect_chorus else 0.005) * sample_rate)
        modulated_signal = np.zeros_like(audio_signal)
        for i in range(len(audio_signal)):
            modulated_delay = int(delay * (1 + np.sin(2 * np.pi * mod_freq * i / sample_rate)))
            if i >= modulated_delay:
                modulated_signal[i] = audio_signal[i - modulated_delay]
        processed_signal += modulated_signal * (0.5 if effect_chorus else 0.7)

    # Plot results if required
    plot_signal(audio_signal, sample_rate, plot_freq, plot_time, title="Original Signal")
    plot_signal(processed_signal, sample_rate, plot_freq, plot_time, title="Processed Signal")

    # Save processed audio if format is specified
    final_save_path = None
    if save_audio_format:
        if save_path is None:
            save_path = os.getcwd()
        final_save_path = os.path.join(save_path, f"processed_audio_effect.{save_audio_format.lower()}")
        if save_audio_format.lower() == "wav":
            wavfile.write(final_save_path, sample_rate, processed_signal.astype(np.int16))
        else:
            audio_segment = AudioSegment(
                processed_signal.tobytes(), frame_rate=sample_rate, sample_width=processed_signal.dtype.itemsize, channels=1
            )
            audio_segment.export(final_save_path, format=save_audio_format.lower())

    return processed_signal, final_save_path

def apply_transformations(audio_signal, sample_rate=44100, apply_fft=True, apply_dft=False, apply_stft=False,
                          plot_freq=False, plot_time=False, save_path=None):
    
    transformed_signal = None
    if apply_fft or apply_dft:
        transformed_signal = fft(audio_signal)
    
    elif apply_stft:
        _, _, transformed_signal = stft(audio_signal, fs=sample_rate)
    
    # Plot and save if needed
    plot_signal(audio_signal, sample_rate, plot_freq, plot_time, transformed_signal=transformed_signal, title="Transformed Signal")

    if save_path:
        np.save(os.path.join(save_path, "transformed_signal.npy"), transformed_signal)
    
    return transformed_signal

def generate_sine_wave(frequency, duration, sample_rate, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def convolution(audio_signal, sine_signal, sample_rate=44100, plot_freq=False, plot_time=False, save_path=None, conv_type="overlap"):
    
    if conv_type == "overlap" and len(audio_signal) < len(sine_signal):
        raise ValueError("Audio signal length must be >= sine signal length for 'overlap' method.")

    segment_length = len(sine_signal)
    convolved_signal = np.zeros(len(audio_signal))

    for i in range(0, len(audio_signal), segment_length):
        segment = audio_signal[i:i + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        
        segment_fft = fft(segment)
        sine_fft = fft(sine_signal, n=segment_length)
        convolved_signal[i:i + segment_length] = np.real(ifft(segment_fft * sine_fft))

    plot_signal(audio_signal, sample_rate, plot_freq, plot_time, title="Original Signal")
    plot_signal(convolved_signal, sample_rate, plot_freq, plot_time, title="Convolved Signal")

    if save_path:
        final_save_path = os.path.join(save_path, "convolved_signal.npy")
        np.save(final_save_path, convolved_signal)
    return convolved_signal

# Filter function using Butterworth filters
def butter_filter(audio_signal, filter_type, cutoff_freqs, fs=44100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = (cutoff_freqs / nyquist) if isinstance(cutoff_freqs, (int, float)) else [f / nyquist for f in cutoff_freqs]
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, audio_signal)

def apply_filters(audio_signal, filters=None, plot_options=None, save_path=None, sample_rate=44100):
    filters = filters or {}
    plot_options = plot_options or {}
    filtered_signal = audio_signal.copy()
    
    if filters.get("lpf"):
        filtered_signal = butter_filter(filtered_signal, 'low', filters['lpf'], fs=sample_rate)
    if filters.get("hpf"):
        filtered_signal = butter_filter(filtered_signal, 'high', filters['hpf'], fs=sample_rate)
    if filters.get("bpf"):
        filtered_signal = butter_filter(filtered_signal, 'bandpass', filters['bpf'], fs=sample_rate)
    if filters.get("apf"):
        filtered_signal = butter_filter(filtered_signal, 'allpass', filters['apf'], fs=sample_rate)

    if plot_options.get("plot_time"):
        plot_signal(filtered_signal, sample_rate, domain="time", title="Filtered Signal (Time Domain)")
    if plot_options.get("plot_freq"):
        plot_signal(filtered_signal, sample_rate, domain="freq", title="Filtered Signal (Frequency Domain)")

    if save_path:
        final_save_path = os.path.join(save_path, "filtered_audio.npy")
        np.save(final_save_path, filtered_signal)
    
    return filtered_signal