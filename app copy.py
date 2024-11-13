from flask import Flask, request, render_template, send_file
import matplotlib.pyplot as plt
import os
import numpy as np

app = Flask(__name__)

# Folder to save temporary plot images
TEMP_FOLDER = "static/temp_plots"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading audio file
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join('uploads', filename))  # Save the uploaded file to the 'uploads' folder
    return render_template('index.html', filename=filename)

# Route for analyzing the audio (e.g., FFT or STFT)
@app.route('/analyze', methods=['POST'])
def analyze():
    filename = request.form['filename']
    analysis_type = request.form['analysis_type']
    
    # Assuming your analysis generates a plot
    plt.figure()
    if analysis_type == 'fft':
        # Example FFT plot
        x = np.linspace(0, 2 * np.pi, 1000)
        y = np.sin(x)  # Example sine wave as the data
        plt.plot(x, y)
        plt.title("FFT Analysis")
    elif analysis_type == 'stft':
        # Example STFT plot (you would replace this with real STFT code)
        plt.plot([0, 1, 2], [1, 2, 3])  # Placeholder for STFT
        plt.title("STFT Analysis")
    elif analysis_type == 'dft':
        # Example DFT plot (you would replace this with real STFT code)
        plt.plot([0, 1, 2], [1, 2, 3])  # Placeholder for STFT
        plt.title("STFT Analysis")

    # Save the plot as a PNG image
    plot_filename = f"{TEMP_FOLDER}/analysis_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    return render_template('index.html', filename=filename, analysis_plot=plot_filename)

# Route for applying filters to the audio
@app.route('/filter', methods=['POST'])
def filter():
    filename = request.form['filename']
    filter_type = request.form['filter_type']
    
    # Assuming your filtering generates a plot
    plt.figure()
    if filter_type == 'lpf':
        # Example Low Pass Filter plot
        plt.plot([0, 1, 2], [3, 2, 1])  # Placeholder for LPF
        plt.title("Low Pass Filter")
    elif filter_type == 'hpf':
        # Example High Pass Filter plot
        plt.plot([0, 1, 2], [1, 2, 3])  # Placeholder for HPF
        plt.title("High Pass Filter")
    elif filter_type == 'bpf':
        # Example Band Pass Filter plot
        plt.plot([0, 1, 2], [1, 3, 2])  # Placeholder for BPF
        plt.title("Band Pass Filter")
    elif filter_type == 'apf':
        # Example All Pass Filter plot
        plt.plot([0, 1, 2], [1, 3, 2])  # Placeholder for BPF
        plt.title("All Pass Filter")
    
    # Save the plot as a PNG image
    plot_filename = f"{TEMP_FOLDER}/filter_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    return render_template('index.html', filename=filename, filter_plot=plot_filename)

# Route for applying convolution to the audio
@app.route('/convolve', methods=['POST'])
def convolve():
    filename = request.form['filename']
    # Example Convolution plot
    plt.figure()
    plt.plot([0, 1, 2], [3, 1, 4])  # Placeholder for Convolution result
    plt.title("Convolution Plot")
    
    # Save the plot as a PNG image
    plot_filename = f"{TEMP_FOLDER}/convolve_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    return render_template('index.html', filename=filename, convolve_plot=plot_filename)

# Route for applying effects to the audio
@app.route('/effects', methods=['POST'])
def effects():
    filename = request.form['filename']
    effect_type = request.form['effect_type']
    
    # Assuming your effects generate a plot
    plt.figure()
    if effect_type == 'reverb':
        # Example Reverb effect plot
        plt.plot([0, 1, 2], [2, 2, 1])  # Placeholder for Reverb effect
        plt.title("Reverb Effect")
    elif effect_type == 'distortion':
        # Example Distortion effect plot
        plt.plot([0, 1, 2], [1, 3, 4])  # Placeholder for Distortion effect
        plt.title("Distortion Effect")

    # Save the plot as a PNG image
    plot_filename = f"{TEMP_FOLDER}/effects_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    return render_template('index.html', filename=filename, effects_plot=plot_filename)

if __name__ == '__main__':
    app.run(debug=True)
