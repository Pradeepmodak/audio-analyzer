import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math

SAMPLE_RATE = 44100
SAMPLE_SIZE = 4096
MONO = 1

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

def get_microphone_list():
    pyaud = pyaudio.PyAudio()
    devices = {}
    for i in range(pyaud.get_device_count()):
        dev_info = pyaud.get_device_info_by_index(i)
        if int(dev_info["maxInputChannels"]) > 0:
            devices[dev_info["name"]] = dev_info["index"]
    pyaud.terminate()
    return devices

def find_peak_frequency_by_derivative(freq_data, magnitude):
    derivative = [magnitude[i + 1] - magnitude[i] for i in range(len(magnitude) - 1)]
    peak_indices = [i for i in range(1, len(derivative)) if derivative[i - 1] > 0 and derivative[i] < 0]

    if not peak_indices:
        return None

    best_peak = max(peak_indices, key=lambda i: magnitude[i])
    return freq_data[best_peak]

def record_audio(mic_index):
    pyaud = pyaudio.PyAudio()

    try:
        stream = pyaud.open(format=pyaudio.paInt16,
                            channels=MONO,
                            rate=SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=SAMPLE_SIZE,
                            input_device_index=mic_index)
    except OSError:
        print("Invalid microphone index. Please select a valid microphone.")
        pyaud.terminate()
        return

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=100)

    time_data = np.arange(SAMPLE_SIZE) / SAMPLE_RATE
    line1, = ax1.plot(time_data, np.zeros(SAMPLE_SIZE), color='b', label="Audio Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time-Domain Waveform")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-32768, 32767)

    freq_data = compute_fft_frequencies(SAMPLE_SIZE, SAMPLE_RATE)
    line2, = ax2.plot(freq_data, np.zeros(len(freq_data)), color='r', label="FFT Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Frequency-Domain Spectrum")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 50000)

    # Initialize text annotation for peak frequency
    text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, color='green', fontsize=12)

    try:
        while True:
            audio_data = np.frombuffer(stream.read(SAMPLE_SIZE, exception_on_overflow=False), dtype=np.int16)
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

    except KeyboardInterrupt:
        print("\nStopping...")
        stream.stop_stream()
        stream.close()
        pyaud.terminate()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    print("Available Microphones:", get_microphone_list())
    try:
        mic_index = int(input("Enter the index of your microphone: "))
        record_audio(mic_index)
    except ValueError:
        print("Invalid input. Please enter a valid microphone index.")

# Resources
# https://builtin.com/articles/fast-fourier-transform
# https://www.dsprelated.com/showarticle/1712.php
# https://www.sweetwater.com/sweetcare/articles/sample-rate/
# Audio Bit Depth and Sample Rate Explained - YouTube  