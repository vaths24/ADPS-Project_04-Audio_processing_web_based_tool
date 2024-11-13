import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import os

def generate_sine_wave(frequency, duration, sample_rate, amplitude=1.0):
    """
    Generates a sine wave signal.

    Parameters:
    frequency (float): Frequency of the sine wave in Hz.
    duration (float): Duration of the sine wave in seconds.
    sample_rate (int): Sample rate of the sine wave in Hz.
    amplitude (float): Amplitude of the sine wave. Default is 1.0.

    Returns:
    numpy array: Generated sine wave signal.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return sine_wave

def convolution(audio_signal, sine_signal, plot_freq=False, plot_time=False, save_path=None, conv_type="overlap", sample_rate=44100):
    """
    Perform convolution between an audio signal and a sine signal using either overlap-save or zero-padding.

    Parameters:
    audio_signal (numpy array): Input audio signal to be convolved.
    sine_signal (numpy array): Sine signal for convolution.
    plot_freq (bool): If True, plots the audio, sine, and convolved signals in the frequency domain. Default is False.
    plot_time (bool): If True, plots the audio, sine, and convolved signals in the time domain. Default is False.
    save_path (str): Path to save the convolved signal. If None, saves in the current directory.
    conv_type (str): Type of convolution to perform - "overlap" or "padding". Default is "overlap".
    sample_rate (int): The sample rate of the signals (default is 44100 Hz).

    Returns:
    convolved_signal (numpy array): Result of the convolution.
    final_save_path (str): The path where the convolved signal is saved.
    """
    # Check if the audio signal is longer than or equal to the sine signal for overlap method
    if conv_type == "overlap" and len(audio_signal) < len(sine_signal):
        raise ValueError("For 'overlap' convolution, the audio signal length must be greater than or equal to the sine signal length.")

    segment_length = len(sine_signal)
    len_audio = len(audio_signal)

    if conv_type == "overlap":
        # Initialize the final convolved signal array
        convolved_signal = np.zeros(len_audio)

        # Process each segment of the audio signal using overlap-save method
        for i in range(0, len_audio, segment_length):
            segment = audio_signal[i:i + segment_length]

            # Zero-pad segment if it's shorter than the sine signal
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))

            # Perform convolution in frequency domain
            segment_fft = fft(segment)
            sine_fft = fft(sine_signal, n=segment_length)
            convolved_segment = np.real(ifft(segment_fft * sine_fft))

            # Store the convolved segment in the output signal with overlap
            end_index = min(i + segment_length, len_audio)
            convolved_signal[i:end_index] += convolved_segment[:end_index - i]

    elif conv_type == "padding":
        # Zero-pad the audio and sine signal for traditional convolution
        padded_audio = np.pad(audio_signal, (0, segment_length - 1))
        padded_sine = np.pad(sine_signal, (0, len(padded_audio) - segment_length))

        # Perform convolution in the frequency domain
        convolved_fft = fft(padded_audio) * fft(padded_sine)
        convolved_signal = np.real(ifft(convolved_fft))

        # Trim the signal to match the original audio length
        convolved_signal = convolved_signal[:len_audio]

    # Plot frequency domain if required
    if plot_freq:
        plt.figure(figsize=(15, 5))

        # Plot the original audio signal in the frequency domain
        plt.subplot(1, 3, 1)
        plt.plot(np.abs(fft(audio_signal)))
        plt.title("Audio Signal (Frequency Domain)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # Plot the sine signal in the frequency domain
        plt.subplot(1, 3, 2)
        plt.plot(np.abs(fft(sine_signal)))
        plt.title("Sine Signal (Frequency Domain)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # Plot the convolved signal in the frequency domain
        plt.subplot(1, 3, 3)
        plt.plot(np.abs(fft(convolved_signal)))
        plt.title("Convolved Signal (Frequency Domain)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        plt.show()

    # Plot time domain if required
    if plot_time:
        time_audio = np.linspace(0, len(audio_signal) / sample_rate, len(audio_signal))
        time_sine = np.linspace(0, len(sine_signal) / sample_rate, len(sine_signal))
        time_convolved = np.linspace(0, len(convolved_signal) / sample_rate, len(convolved_signal))

        plt.figure(figsize=(15, 5))

        # Plot the original audio signal in the time domain
        plt.subplot(1, 3, 1)
        plt.plot(time_audio, audio_signal)
        plt.title("Audio Signal (Time Domain)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # Plot the sine signal in the time domain
        plt.subplot(1, 3, 2)
        plt.plot(time_sine, sine_signal)
        plt.title("Sine Signal (Time Domain)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # Plot the convolved signal in the time domain
        plt.subplot(1, 3, 3)
        plt.plot(time_convolved, convolved_signal)
        plt.title("Convolved Signal (Time Domain)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

    # Determine save path
    if save_path is None:
        save_path = os.getcwd()
    final_save_path = os.path.join(save_path, "convolved_signal.npy")
    np.save(final_save_path, convolved_signal)

    return convolved_signal, final_save_path

# Generate the sine wave signal
frequency = 20  # Frequency of the sine wave in Hz (e.g., A4 note)
duration = 1.0   # Duration in seconds
sample_rate = 44100  # Sampling rate in Hz
amplitude = 0.5  # Amplitude of the sine wave

sine_signal = generate_sine_wave(frequency, duration, sample_rate, amplitude)

# Example audio signal (replace with actual data as needed)
audio_signal = np.random.randn(sample_rate * 2)  # 2 seconds of random audio at 44.1 kHz

# Call the convolution function with the chosen method and time-domain plotting enabled
convolved_signal, save_path = convolution(audio_signal, sine_signal, plot_freq=True, plot_time=True, conv_type="overlap")
print("Convolved signal saved at:", save_path)
