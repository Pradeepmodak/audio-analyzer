import librosa as lr  # Python module for audio processing
import numpy as np  # Python module for scientific computing
import matplotlib.pyplot as plt  # Python module for plotting
from glob2 import glob  # Python module for finding files

# Function to plot the tempo and rhythm of the audio signal
def plot_tempo_rhythm(audio, sfreq, ax):
    """
    Plots the waveform of the audio with beat markers.

    Parameters:
    - audio: Audio time-series data
    - sfreq: Sampling frequency
    - ax: Matplotlib axis object for plotting
    """
    tempo, beat_frames = lr.beat.beat_track(y=audio, sr=sfreq)
    beat_times = lr.frames_to_time(beat_frames, sr=sfreq)

    ax.plot(np.arange(len(audio)) / sfreq, audio, alpha=0.6, color='b')
    ax.vlines(beat_times, ymin=-1, ymax=1, color='r', linestyle='--', label="Beats")
    ax.set_title("Tempo and Rhythm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()

# Function to plot the amplitude (waveform) of the audio signal
def plot_amplitude(audio, sfreq, ax):
    """
    Plots the amplitude of the audio signal over time.

    Parameters:
    - audio: Audio time-series data
    - sfreq: Sampling frequency
    - ax: Matplotlib axis object for plotting
    """
    time = np.arange(len(audio)) / sfreq
    ax.plot(time, audio, color='g')
    ax.set_title("Amplitude Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sound Amplitude")

# Function to plot the pitch contour of the audio
def plot_pitch(audio, sfreq, ax):
    """
    Extracts and plots the pitch contour of the audio signal.

    Parameters:
    - audio: Audio time-series data
    - sfreq: Sampling frequency
    - ax: Matplotlib axis object for plotting
    """
    pitches, _ = lr.piptrack(y=audio, sr=sfreq)
    pitch_values = np.max(pitches, axis=0)  # Extract the highest pitch values at each time step
    time = np.linspace(0, len(audio) / sfreq, num=pitch_values.shape[0])

    ax.plot(time, pitch_values, label="Pitch (Hz)", color='r')
    ax.set_title("Pitch Contour")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.legend()

# Function to plot the magnitude contour of the audio
def plot_magnitude(audio, sfreq, ax):
    """
    Extracts and plots the magnitude contour of the audio signal.

    Parameters:
    - audio: Audio time-series data
    - sfreq: Sampling frequency
    - ax: Matplotlib axis object for plotting
    """
    _, magnitudes = lr.piptrack(y=audio, sr=sfreq)
    mag_values = np.max(magnitudes, axis=0)  # Extract the highest magnitude at each time step
    time = np.linspace(0, len(audio) / sfreq, num=mag_values.shape[0])

    ax.plot(time, mag_values, label="Magnitude", color='m')
    ax.set_title("Magnitude Contour")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.legend()

# Function to plot all the graphs in a 2x2 grid
def plot_all(audio, sfreq):
    """
    Plots all four audio analysis features (tempo, amplitude, pitch, magnitude)
    in a single figure with a 2x2 grid layout.

    Parameters:
    - audio: Audio time-series data
    - sfreq: Sampling frequency
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Creating a 2x2 grid layout

    plot_tempo_rhythm(audio, sfreq, axes[0, 0])
    plot_amplitude(audio, sfreq, axes[0, 1])
    plot_pitch(audio, sfreq, axes[1, 0])
    plot_magnitude(audio, sfreq, axes[1, 1])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

# Load audio files
audio_files = glob('./assets/*.wav')  # Find all .wav files in the assets folder
audio, sfreq = lr.load(audio_files[1])  # Load the second audio file from the list

# Call the function to display all graphs in a 2x2 grid layout
plot_all(audio, sfreq)
