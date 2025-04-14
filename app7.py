import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import wave
import os

SAMPLE_RATE = 44100
SAMPLE_SIZE = 4096
MONO = 1

AUDIO_FOLDER = "audiosamples"

def compute_fft(signal):
    N = len(signal)
    if N <= 1:
        return signal

    even_fft = compute_fft(signal[0::2])
    odd_fft = compute_fft(signal[1::2])

    T = [complex(math.cos(-2 * math.pi * k / N), math.sin(-2 * math.pi * k / N)) * odd_fft[k] for k in range(N // 2)]

    return [even_fft[k] + T[k] for k in range(N // 2)] + [even_fft[k] - T[k] for k in range(N // 2)]

def compute_fft_frequencies(N, sample_rate):
    return [(k * sample_rate) / N for k in range(N // 2)]

def find_peak_frequency_by_derivative(freq_data, magnitude):
    derivative = [magnitude[i + 1] - magnitude[i] for i in range(len(magnitude) - 1)]
    peak_indices = [i for i in range(1, len(derivative)) if derivative[i - 1] > 0 and derivative[i] < 0]

    if not peak_indices:
        return None

    best_peak = max(peak_indices, key=lambda i: magnitude[i])
    return freq_data[best_peak]

def visualize_fft_stream(audio_data_stream):
    plt.ion()
    fig = plt.figure(figsize=(14, 7), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    fig.suptitle("Audio FFT Visualization", fontsize=16, fontweight='bold')

    ax1 = plt.subplot(gs[0])
    time_data = np.arange(SAMPLE_SIZE) / SAMPLE_RATE
    line1, = ax1.plot(time_data, np.zeros(SAMPLE_SIZE), color='#1f77b4', linewidth=1.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time-Domain Waveform")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(-32768, 32767)

    ax2 = plt.subplot(gs[1])
    freq_data = compute_fft_frequencies(SAMPLE_SIZE, SAMPLE_RATE)
    line2, = ax2.plot(freq_data, np.zeros(len(freq_data)), color='#d62728', linewidth=1.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Frequency-Domain Spectrum")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 10000)

    text = ax2.text(0.7, 0.85, '', transform=ax2.transAxes, color='green', fontsize=12, bbox=dict(facecolor='white', edgecolor='green'))

    for audio_data in audio_data_stream:
        if len(audio_data) != SAMPLE_SIZE:
            continue

        fft_complex = np.array(compute_fft(audio_data))
        fft_magnitude = np.abs(fft_complex[:SAMPLE_SIZE // 2])

        peak_freq = find_peak_frequency_by_derivative(freq_data, fft_magnitude)
        if peak_freq:
            print(f"Dominant Frequency (derivative): {peak_freq:.2f} Hz")
            text.set_text(f"Peak Frequency: {peak_freq:.2f} Hz")

        line1.set_ydata(audio_data)
        line2.set_ydata(fft_magnitude)

        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def process_audio_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data_stream = []

        while True:
            frames = wf.readframes(SAMPLE_SIZE)
            if not frames or len(frames) < SAMPLE_SIZE * 2:
                break
            audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_data_stream.append(audio_data)

        if audio_data_stream:
            visualize_fft_stream(audio_data_stream)

if __name__ == "__main__":
    folder_path = AUDIO_FOLDER
    if os.path.isdir(folder_path):
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not wav_files:
            print("No .wav files found in the directory.")
        else:
            file_name = wav_files[4]  # Only process the first one
            print(f"Processing: {file_name}")
            process_audio_file(os.path.join(folder_path, file_name))
    else:
        print("Invalid folder path.")