import librosa as lr # python module for audio processing
import numpy as np # python module for scientific computing
import matplotlib.pyplot as plt # python module for plotting
from glob2 import glob # python module for finding files

def plot_tempo_rhythm(audio, sfreq):
    
    # Compute tempo and beat frames
    tempo, beat_frames = lr.beat.beat_track(y=audio, sr=sfreq)

    # Convert beat frames to time
    beat_times = lr.frames_to_time(beat_frames, sr=sfreq)

    # Plot waveform with beat markers
    plt.figure(figsize=(10, 4))
    lr.display.waveshow(audio, sr=sfreq, alpha=0.6)
    plt.vlines(beat_times, ymin=-1, ymax=1, color='r', linestyle='--', label="Beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Tempo and Rhythm of Audio")
    plt.legend()
    plt.show()


def plot_amplitude(audio, sfreq):
    time=np.arange(len(audio))/sfreq
    fig, ax = plt.subplots(figsize=(10, 4)) 
    ax.plot(time, audio, color='b')
    ax.set(xlabel='Time (s)',ylabel='Sound Amplitude')
    plt.show()

def plot_pitch(audio, sfreq):

    # Extract pitch using librosa
    pitches,_ = lr.piptrack(y=audio, sr=sfreq)

    # Convert to fundamental frequency
    pitch_values = np.max(pitches, axis=0)  

    # Time axis
    time = np.linspace(0, len(audio) / sfreq, num=pitch_values.shape[0])

    # Plot pitch contour
    plt.figure(figsize=(10, 4))
    plt.plot(time, pitch_values, label="Pitch (Hz)", color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("PItch (Hz)")
    plt.title("Pitch Contour of Audio")
    plt.legend()
    plt.show()

def plot_magnitude(audio, sfreq):

    # Extract pitch using librosa
    _, magnitudes = lr.piptrack(y=audio, sr=sfreq)

    # Convert to fundamental frequency
    mag_values = np.max(magnitudes, axis=0)

    # Time axis
    time = np.linspace(0, len(audio) / sfreq, num=mag_values.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(time, mag_values, label="Magnitude", color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.title("Magnitude Contour of Audio")
    plt.legend()
    plt.show()

audio_files = glob('./assets/*.wav') # adding audio files

audio, sfreq =lr.load(audio_files[1])
plot_tempo_rhythm(audio,sfreq)
plot_amplitude(audio,sfreq)
plot_pitch(audio,sfreq)
plot_magnitude(audio,sfreq)
