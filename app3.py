import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time

class APU:
    MONO = 1
    DEFAULT_SAMPLE_SIZE = 1024


    def __init__(self):
        self.__pyaud = pyaudio.PyAudio()
        self.sample_size = self.DEFAULT_SAMPLE_SIZE
        self.stream = None
        self.mic_index = None
        self.sampling_rate = 44100  # Standard sampling rate

    def getMicrophoneList(self):
        """Returns a dictionary of available microphone devices."""
        retval = {}
        for devindex in range(self.__pyaud.get_device_count()):
            adict = self.__pyaud.get_device_info_by_index(devindex)
            if int(adict["maxInputChannels"]) > 0:
                retval[adict["name"]] = adict["index"]
        return retval

    def setMicrophone(self, index):
        """Sets the microphone index and updates the sampling rate."""
        self.mic_index = index
        if self.stream:
            self.stop()
        self.sampling_rate = int(self.__pyaud.get_device_info_by_index(self.mic_index)["defaultSampleRate"])

    def calcFrequency(self):
        """Reads audio data, applies FFT, and returns the peak frequency."""
        if not self.stream:
            return 0.0

        readbuf = self.stream.read(self.sample_size, exception_on_overflow=False)
        tdarray = np.frombuffer(readbuf, dtype=np.int16)  # Correct dtype

        # Apply FFT
        fftarray = np.abs(np.fft.fft(tdarray))[:self.sample_size // 2]
        freqarray = np.fft.fftfreq(self.sample_size, d=1/self.sampling_rate)[:self.sample_size // 2]

        # Find peak frequency
        peak_index = np.argmax(fftarray)
        peak_freq = freqarray[peak_index]

        return abs(peak_freq)

    def start(self):
        """Starts audio streaming."""
        if self.stream:
            self.stop()

        if self.mic_index is None:
            self.mic_index = int(self.__pyaud.get_default_input_device_info()["index"])
        if not self.sampling_rate:
            self.sampling_rate = int(self.__pyaud.get_device_info_by_index(self.mic_index)["defaultSampleRate"])

        self.stream = self.__pyaud.open(format=pyaudio.paInt16,
                                        channels=self.MONO,
                                        input_device_index=self.mic_index,
                                        rate=self.sampling_rate,
                                        input=True,
                                        frames_per_buffer=self.sample_size)

    def stop(self):
        """Stops and closes the audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def implode(self):
        """Stops audio processing and terminates PyAudio."""
        self.stop()
        self.__pyaud.terminate()

    def test(self):
        """Continuously detects and plots the frequency in real time with note labels."""
        self.start()

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))

        x_data, y_data, note_labels = [], [], []
        line, = ax.plot(x_data, y_data, marker="o", linestyle="-", color="b", markersize=5, label="Detected Frequency")
        text_annotations = []

        ax.set_ylim(0, 1000)  # Frequency range (Hz)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Frequency (Hz)", fontsize=14)
        ax.set_title("Real-Time Frequency Plot", fontsize=16, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=12)

        start_time = time.time()

        try:
            while True:
                frequency = self.calcFrequency()
                elapsed_time = time.time() - start_time

                print(f"Detected Frequency: {frequency:.2f} Hz")

                x_data.append(elapsed_time)
                y_data.append(frequency)

                if len(x_data) > 50:
                    x_data.pop(0)
                    y_data.pop(0)

                line.set_xdata(x_data)
                line.set_ydata(y_data)

                ax.set_xlim(max(0, elapsed_time - 10), elapsed_time + 1)
                ax.relim()
                ax.autoscale_view()

                # Remove old annotations
                for txt in text_annotations:
                    txt.remove()
                text_annotations.clear()

                # Add new annotations
                for i, (x, y, label) in enumerate(zip(x_data, y_data, note_labels)):
                    if label != "Unknown":
                        txt = ax.text(x, y, label, fontsize=10, ha='right', color='red')
                        text_annotations.append(txt)

                plt.draw()
                plt.pause(0.1)

        except KeyboardInterrupt:
            print("\nStopping...")
            self.implode()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    apu = APU()
    print("Available Microphones:", apu.getMicrophoneList())

    mic_index = int(input("Enter the index of your microphone: "))
    apu.setMicrophone(mic_index)

    apu.test()
