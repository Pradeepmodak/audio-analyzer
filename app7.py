import pyaudio
# to handle audio input/output
import numpy as np
# for numerical operations
import matplotlib.pyplot as plt
# for plotting
import math
# for mathematical functions

# Constants
SAMPLE_RATE = 44100  # Standard audio sampling rate
# Har second mein 44,100 samples liye ja rahe hain.
# Number of samples per frame
SAMPLE_SIZE = 4096   # Higher sample size for smoother plots
# Jab tum audio signal ko sample karte ho (sampling rate se), 
# tab har sample ko digitally store karne ke liye ek number chahiye 
# — aur sample size batata hai us number ka resolution.
MONO = 1
# This means the microphone records in mono (single channel)
#mtlb audio me koi direction nhi hoga jaise left ya right
# speech aur voice cover ke lie best h

def compute_fft(signal):
    # audio signal ya waveform ko input leta hai 
    # — aur uska frequency domain representation 
    # nikalta hai using FFT.
    # Computes the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm (recursive).
    # This version explicitly uses sine and cosine functions instead of `exp(jθ)`.
    # :param signal: Input time-domain signal (list of real numbers)
    # :return: FFT spectrum (list of complex numbers)
    N = len(signal)
# Socho tumne ek 6-second ka voice message record kiya, jisme 1 second mein 1 value store ho rahi hai.
# Toh signal mein total 6 points honge, aur len(signal) → 6.
    if N <= 1:
        return signal

    # Divide: Even and Odd indexed elements
    even_fft = compute_fft(signal[0::2])
    odd_fft = compute_fft(signal[1::2])

    # Compute FFT using sine and cosine
    T = [complex(math.cos(-2 * math.pi * k / N), math.sin(-2 * math.pi * k / N)) * odd_fft[k] for k in range(N // 2)]

    return [even_fft[k] + T[k] for k in range(N // 2)] + [even_fft[k] - T[k] for k in range(N // 2)]

def compute_fft_frequencies(N, sample_rate):
    # Computes the FFT frequency bins 
    # :param N: Number of samples (FFT length)
    # :param sample_rate: Sampling rate of the audio signal
    # :return: Array of frequency bins
    return [(k * sample_rate) / N for k in range(N // 2)]
    # N // 2 ka matlab hai: N ko half kar ke sequence banayi jaayegi
    # Yeh jo N // 2 use hota hai, uska reason actually bahut simple hai — jab tum FFT perform karte ho,
    #  toh tumhe sirf positive frequencies ki zarurat hoti hai.

def get_microphone_list():
    #Returns a dictionary of available microphone devices.
    pyaud = pyaudio.PyAudio()
    devices = {}
    for i in range(pyaud.get_device_count()):
        dev_info = pyaud.get_device_info_by_index(i)
        if int(dev_info["maxInputChannels"]) > 0:
            devices[dev_info["name"]] = dev_info["index"]
    pyaud.terminate()
    return devices

def record_audio(mic_index):
    #Records audio from the selected microphone and plots waveforms in real-time.
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

    # Time-domain plot (Amplitude vs Time)
    time_data = np.arange(SAMPLE_SIZE) / SAMPLE_RATE
    line1, = ax1.plot(time_data, np.zeros(SAMPLE_SIZE), color='b', label="Audio Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time-Domain Waveform")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-32768, 32767)  # 16-bit integer range

    # Frequency-domain plot (Frequency vs Magnitude)
    freq_data = compute_fft_frequencies(SAMPLE_SIZE, SAMPLE_RATE)
    line2, = ax2.plot(freq_data, np.zeros(len(freq_data)), color='r', label="FFT Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Frequency-Domain Spectrum")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 50000)  # Set upper limit for better visualization

    try:
        while True:
            # Read audio data
            audio_data = np.frombuffer(stream.read(SAMPLE_SIZE, exception_on_overflow=False), dtype=np.int16)

            # Compute FFT manually (convert result to NumPy array)
            fft_complex = np.array(compute_fft(audio_data))
            fft_magnitude = np.abs(fft_complex[:SAMPLE_SIZE // 2])  # Take half for positive frequencies

            # Update time-domain plot
            line1.set_ydata(audio_data)

            # Update frequency-domain plot
            line2.set_ydata(fft_magnitude)

            # Update plots
            plt.draw()
            plt.pause(0.05)  # Faster updates for a smoother real-time effect

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


# resouces 
# https://builtin.com/articles/fast-fourier-transform
# https://www.dsprelated.com/showarticle/1712.php
# https://www.sweetwater.com/sweetcare/articles/sample-rate/
# Audio Bit Depth and Sample Rate Explained - YouTube