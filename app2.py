import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile  # Get the API
import sounddevice as sd
import librosa
import plotly.graph_objects as go
import ffmpeg  # Ensure FFmpeg is installed

# Install missing dependencies
try:
    import pyaudio
except ImportError:
    print("Installing missing dependencies...")
    os.system("pip install numpy scipy matplotlib librosa tqdm plotly pyaudio sounddevice ffmpeg-python")

# Configuration
FPS = 30
FFT_WINDOW_SECONDS = 0.25  # How many seconds of audio make up an FFT window
FREQ_MIN = 10
FREQ_MAX = 1000
TOP_NOTES = 3
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
RESOLUTION = (1920, 1080)
SCALE = 2  # Adjust for higher resolution

# File Path (Modify as per your local system)
AUDIO_FILE = "piano_c_major_scale.wav"  # Place the file in the same folder as this script

# Ensure the file exists
if not os.path.exists(AUDIO_FILE):
    print(f"Audio file {AUDIO_FILE} not found! Downloading...")
    os.system("wget https://github.com/jeffheaton/present/raw/master/youtube/video/sample_audio/piano_c_major_scale.wav")

fs, data = wavfile.read(AUDIO_FILE)  # Load the data
audio = data.T[0]  # Extract first channel if stereo
FRAME_STEP = (fs / FPS)  # Audio samples per video frame
FFT_WINDOW_SIZE = int(fs * FFT_WINDOW_SECONDS)
AUDIO_LENGTH = len(audio) / fs
FRAME_COUNT = int(AUDIO_LENGTH * FPS)
FRAME_OFFSET = int(len(audio) / FRAME_COUNT)

# Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, FFT_WINDOW_SIZE, False)))

xf = np.fft.rfftfreq(FFT_WINDOW_SIZE, 1 / fs)


# Convert frequency to musical note
def freq_to_number(f):
    return 69 + 12 * np.log2(f / 440.0)


def number_to_freq(n):
    return 440 * 2.0 ** ((n - 69) / 12.0)


def note_name(n):
    return NOTE_NAMES[n % 12] + str(int(n / 12 - 1))


def plot_fft(p, xf, fs, notes, dimensions=(960, 540)):
    layout = go.Layout(
        title="Frequency Spectrum",
        autosize=False,
        width=dimensions[0],
        height=dimensions[1],
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        font={'size': 24}
    )

    fig = go.Figure(layout=layout,
                    layout_xaxis_range=[FREQ_MIN, FREQ_MAX],
                    layout_yaxis_range=[0, 1])

    fig.add_trace(go.Scatter(x=xf, y=p))

    for note in notes:
        fig.add_annotation(x=note[0] + 10, y=note[2],
                           text=note[1],
                           font={'size': 48},
                           showarrow=False)
    return fig


def extract_sample(audio, frame_number):
    end = frame_number * FRAME_OFFSET
    begin = int(end - FFT_WINDOW_SIZE)

    if end == 0:
        return np.zeros((np.abs(begin)), dtype=float)
    elif begin < 0:
        return np.concatenate([np.zeros((np.abs(begin)), dtype=float), audio[0:end]])
    else:
        return audio[begin:end]


def find_top_notes(fft_data, num):
    if np.max(fft_data.real) < 0.001:
        return []

    lst = [x for x in enumerate(fft_data.real)]
    lst = sorted(lst, key=lambda x: x[1], reverse=True)

    idx = 0
    found = []
    found_note = set()
    while (idx < len(lst)) and (len(found) < num):
        f = xf[lst[idx][0]]
        y = lst[idx][1]
        n = freq_to_number(f)
        n0 = int(round(n))
        name = note_name(n0)

        if name not in found_note:
            found_note.add(name)
            s = [f, name, y]
            found.append(s)
        idx += 1

    return found


# Pass 1: Find max amplitude for scaling
mx = 0
for frame_number in range(FRAME_COUNT):
    sample = extract_sample(audio, frame_number)
    fft_data = np.fft.rfft(sample * window)
    fft_data = np.abs(fft_data).real
    mx = max(np.max(fft_data), mx)

print(f"Max amplitude: {mx}")

# Pass 2: Generate frames and save images
for frame_number in tqdm.tqdm(range(FRAME_COUNT)):
    sample = extract_sample(audio, frame_number)
    fft_data = np.fft.rfft(sample * window)
    fft_data = np.abs(fft_data) / mx

    s = find_top_notes(fft_data, TOP_NOTES)
    fig = plot_fft(fft_data.real, xf, fs, s, RESOLUTION)
    fig.write_image(f"frame{frame_number}.png", scale=2)

# Combine frames into a video
print("Generating video...")
os.system(f"ffmpeg -y -r {FPS} -f image2 -s 1920x1080 -i frame%d.png -i {AUDIO_FILE} -c:v libx264 -pix_fmt yuv420p movie.mp4")

print("Video saved as movie.mp4")

# Play the generated video (Windows only)
if os.name == "nt":
    os.system("start movie.mp4")
