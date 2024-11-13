from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
from scipy.io.wavfile import write
from pydub import AudioSegment
from io import BytesIO
import matplotlib.pyplot as plt
from scripts.all_filters import apply_audio_effects, apply_transformations, generate_sine_wave, convolution, apply_filters


# Flask App Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'aac'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading and displaying the audio file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html', filename=filename)
    return redirect(url_for('index'))

# Route for signal analysis (FFT, DFT, STFT)
@app.route('/analyze', methods=['POST'])
def analyze():
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_data, sr = librosa.load(filepath)
    analysis_type = request.form['analysis_type']
    
    # Perform FFT, DFT, or STFT
    if analysis_type == 'fft':
        fft_result = np.fft.fft(audio_data)
        result = np.abs(fft_result)
    elif analysis_type == 'stft':
        result = np.abs(librosa.stft(audio_data))
    
    # Generate plot for the result
    plt.figure(figsize=(10, 4))
    if analysis_type == 'stft':
        librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    else:
        plt.plot(result)
    plt.title(f'{analysis_type.upper()} Analysis')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

# Route for filtering (LPF, HPF, BPF)
@app.route('/filter', methods=['POST'])
def apply_filter():
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_data, sr = librosa.load(filepath)
    filter_type = request.form['filter_type']
    
    # Define filters
    nyquist = sr / 2
    if filter_type == 'lpf':
        sos = signal.butter(10, 1000 / nyquist, btype='low', output='sos')
    elif filter_type == 'hpf':
        sos = signal.butter(10, 500 / nyquist, btype='high', output='sos')
    elif filter_type == 'bpf':
        sos = signal.butter(10, [500 / nyquist, 2000 / nyquist], btype='bandpass', output='sos')
    
    filtered_data = signal.sosfilt(sos, audio_data)
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"filtered_{filename}")
    write(output_filepath, sr, (filtered_data * 32767).astype(np.int16))
    
    return send_file(output_filepath, as_attachment=True)

# Route for convolution
@app.route('/convolve', methods=['POST'])
def convolve():
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_data, sr = librosa.load(filepath)
    
    # Simple convolution with impulse response (e.g., reverb effect)
    ir = np.random.normal(0, 1, 2048)  # Example impulse response
    convolved_data = np.convolve(audio_data, ir, mode='same')
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"convolved_{filename}")
    write(output_filepath, sr, (convolved_data * 32767).astype(np.int16))
    
    return send_file(output_filepath, as_attachment=True)

# Route for audio effects 
@app.route('/effects', methods=['POST'])
def apply_effect():
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_data, sr = librosa.load(filepath)
    effect_type = request.form['effect_type']
    
    # Apply effects
    if effect_type == 'echo':
        echo_data = np.concatenate([audio_data, np.zeros(sr)]) + 0.5 * np.concatenate([np.zeros(sr), audio_data])
        output_data = echo_data[:len(audio_data)]
    elif effect_type == 'pitch':
        output_data = librosa.effects.pitch_shift(audio_data, sr, n_steps=2)
    elif effect_type == 'frequency':
        output_data = librosa.effects.time_stretch(audio_data, rate=0.8)
    
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{effect_type}_{filename}")
    write(output_filepath, sr, (output_data * 32767).astype(np.int16))
    
    return send_file(output_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
