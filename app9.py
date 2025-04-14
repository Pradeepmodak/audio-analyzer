import numpy as np
import matplotlib.pyplot as plt
import math
import wave

SAMPLE_RATE = 44100
SAMPLE_SIZE = 4096

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

def convert_to_mono(audio_data, num_channels):
    if num_channels == 1:
        return audio_data
    # For stereo, average the two channels
    return np.mean(audio_data.reshape(-1, num_channels), axis=1).astype(np.int16)

def process_audio_file(filename):
    try:
        wf = wave.open(filename, 'rb')
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except Exception as e:
        print(f"Error opening file: {str(e)}")
        return
    
    num_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    sample_rate = wf.getframerate()
    
    if sample_width != 2:  # 16-bit audio
        print("Error: Audio file must be 16-bit.")
        wf.close()
        return
    
    if sample_rate != SAMPLE_RATE:
        print(f"Warning: File sample rate ({sample_rate}) doesn't match expected rate ({SAMPLE_RATE}). Results may vary.")
    
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

    text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, color='green', fontsize=12)

    try:
        pos = 0
        while True:
            wf.setpos(pos)
            frames = wf.readframes(SAMPLE_SIZE)
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            if len(audio_data) == 0:  # End of file
                break
                
            # Convert to mono if needed
            audio_data = convert_to_mono(audio_data, num_channels)
            
            # Pad with zeros if we didn't get a full buffer
            if len(audio_data) < SAMPLE_SIZE:
                audio_data = np.pad(audio_data, (0, SAMPLE_SIZE - len(audio_data)), 'constant')
            
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
            
            pos += SAMPLE_SIZE
            
            if pos * 2 >= wf.getnframes():  # Multiply by 2 because frames are 16-bit (2 bytes)
                pos = 0  # Loop back to start

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        wf.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    filename = "audiosample.wav"  # You can change this to your file name
    process_audio_file(filename)