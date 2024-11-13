from flask import Flask, request, render_template, send_file
import matplotlib.pyplot as plt
import os
import numpy as np

app = Flask(__name__)

# Folder to save temporary plot images and uploaded files
UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "static/temp_plots"
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_audio_files():
    # Define the audio file extensions to look for
    audio_extensions = ('.wav', '.mp3', '.aac')
    # List comprehension to get files that end with the specified extensions
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(audio_extensions)]

# Route for the homepage
@app.route('/')
def index():
    audio_files = get_audio_files()  # Get list of audio files in 'uploads' folder
    return render_template('index.html', audio_files=audio_files)

# Route for uploading audio file
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(UPLOAD_FOLDER, filename))  # Save uploaded file
    audio_files = get_audio_files()  # Update file list after upload
    return render_template('index.html', audio_files=audio_files, filename=filename)

# Route for analyzing the audio (e.g., FFT or STFT)
@app.route('/analyze', methods=['POST'])
def analyze():
    filename = request.form['filename']
    analysis_type = request.form['analysis_type']
    
    # Generate analysis plot based on analysis_type
    plt.figure()
    if analysis_type == 'fft':
        x = np.linspace(0, 2 * np.pi, 1000)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title("FFT Analysis")
    elif analysis_type == 'stft':
        plt.plot([0, 1, 2], [1, 2, 3])
        plt.title("STFT Analysis")
    elif analysis_type == 'dft':
        plt.plot([0, 1, 2], [1, 2, 3])
        plt.title("DFT Analysis")

    plot_filename = f"{TEMP_FOLDER}/analysis_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    audio_files = get_audio_files()
    return render_template('index.html', audio_files=audio_files, filename=filename, analysis_plot=plot_filename)

# Route for applying filters to the audio
@app.route('/filter', methods=['POST'])
def filter():
    filename = request.form['filename']
    filter_type = request.form['filter_type']
    
    # Generate filter plot
    plt.figure()
    if filter_type == 'lpf':
        plt.plot([0, 1, 2], [3, 2, 1])
        plt.title("Low Pass Filter")
    elif filter_type == 'hpf':
        plt.plot([0, 1, 2], [1, 2, 3])
        plt.title("High Pass Filter")
    elif filter_type == 'bpf':
        plt.plot([0, 1, 2], [1, 3, 2])
        plt.title("Band Pass Filter")
    elif filter_type == 'apf':
        plt.plot([0, 1, 2], [1, 3, 2])
        plt.title("All Pass Filter")
    
    plot_filename = f"{TEMP_FOLDER}/filter_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    audio_files = get_audio_files()
    return render_template('index.html', audio_files=audio_files, filename=filename, filter_plot=plot_filename)

# Route for applying convolution to the audio
@app.route('/convolve', methods=['POST'])
def convolve():
    filename = request.form['filename']
    plt.figure()
    plt.plot([0, 1, 2], [3, 1, 4])
    plt.title("Convolution Plot")
    
    plot_filename = f"{TEMP_FOLDER}/convolve_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    audio_files = get_audio_files()
    return render_template('index.html', audio_files=audio_files, filename=filename, convolve_plot=plot_filename)

# Route for applying effects to the audio
@app.route('/effects', methods=['POST'])
def effects():
    filename = request.form['filename']
    effect_type = request.form['effect_type']
    
    # Generate effects plot
    plt.figure()
    if effect_type == 'reverb':
        plt.plot([0, 1, 2], [2, 2, 1])
        plt.title("Reverb Effect")
    elif effect_type == 'distortion':
        plt.plot([0, 1, 2], [1, 3, 4])
        plt.title("Distortion Effect")

    plot_filename = f"{TEMP_FOLDER}/effects_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    audio_files = get_audio_files()
    return render_template('index.html', audio_files=audio_files, filename=filename, effects_plot=plot_filename)

if __name__ == '__main__':
    app.run(debug=True)
