import numpy as np  #Yeh numerical ke lie hai—arrays,maths sab easy banata hai
import matplotlib.pyplot as plt  #Graph banane ke liye,isse hum spectrogram plot karenge
import math  #Basic maths jaise log,sin,etc. ke liye
import wave  #.wav file ko read karne ke liye — audio ko uthata hai file se
from scipy.signal import spectrogram  #Spectrogram banane ka kaam karta hai — matlab frequency vs time ka visual


#Constant parameters
SAMPLE_RATE = 44100  #Ek second me kitne samples hain—CD quality audio ka standard rate hota hai (44.1kHz)
FFT_SIZE = 4096      #Har baar kitne samples pe FFT lagana hai—zyada size = zyada frequency detail
HOP_SIZE = 512       #Ek FFT ke baad agle FFT me kitna aage badhna hai—chhoti hop = smooth spectrogram


def compute_fft(signal):
    N=len(signal)
    if(N<=1):    #base case:agar signal ek point ka hai, toh uska FFT wahi hai.Recursion yahin ruk jaata hai. 
        return signal
    even_fft=compute_fft(signal[0::2]) #even index wale samples(0,2,4..)
    odd_fft=compute_fft(signal[1::2])  #odd index wale samples(1,3,5..)
    T=[complex(math.cos(-2*math.pi*k/N),math.sin(-2*math.pi*k/N))*odd_fft[k] for k in range(N//2)]
    return [even_fft[k]+T[k] for k in range(N//2)]+[even_fft[k]-T[k]for k in range(N//2)]

def find_peak_frequency_by_derivative(freq_data,magnitude):
    derivative=[magnitude[i+1]-magnitude[i] for i in range(len(magnitude)-1)]
    peak_indices=[i for i in range(1,len(derivative)) if derivative[i-1]>0 and derivative[i]<0]
    if not peak_indices: #Jaha pe derivative pehle positive tha aur baad me negative ho gaya, tohWaha ek local peak hai!
        return None
    best_peak=max(peak_indices,key=lambda i: magnitude[i])
    return freq_data[best_peak]

def analyze_sweep(filename):
    try:
        #to load the audio file
        with wave.open(filename,'rb') as wf: #rb->matlab hai read in binary mode(kyunki audio data raw binary hota hai)  wf->object bn gaya
            params = wf.getparams() #isse hume file ke basic info milta hai: jaise number of channels,sample width,framerate,etc.
            print(f"Audio file parameters: {params}")
            audio_data=np.frombuffer(wf.readframes(params.nframes),dtype=np.int16) #poore audio ke frames read kar leta hai(binary data). aur as a arrau store kr leta hai
        
            #convert to mono if needed m=(l+r)/2
            if params.nchannels>1:
                audio_data=audio_data.reshape(-1,params.nchannels).mean(axis=1)
            
            #apply window function to entire audio data
            window = np.hanning(FFT_SIZE)
            
            #create figure
            plt.figure(figsize=(14,10))
            plt.suptitle(f"Frequency Analysis:{filename}", y=0.98, fontsize=14)
            
            #1.Spectrogram(scipy for efficiency)
            ax1=plt.subplot(2,1,1)
            f,t,Sxx=spectrogram(
                audio_data,
                fs=params.framerate,
                nperseg=FFT_SIZE,
                noverlap=FFT_SIZE-HOP_SIZE,
                window='hann'
            )
            plt.pcolormesh(t,f/1000,10*np.log10(Sxx),  # Removed the spec variable since we're not using colorbar
                             shading='auto', cmap='viridis')
            plt.ylim(0,20)
            plt.ylabel("Frequency (kHz)")
            plt.title("Spectrogram (Time-Frequency Analysis)")
            plt.grid(True, alpha=0.3)
            
            #2.FFT analysis
            ax2=plt.subplot(2, 1, 2)
            analysis_points=np.linspace(0.1,0.9,5)    #audio ke 0.1 to 0.9 (10% to 90%) ke beech se 5 points
            time_points=(analysis_points*len(audio_data)).astype(int)
            colors=plt.cm.plasma(np.linspace(0,1,len(time_points)))
            
            for i,t in enumerate(time_points):
                segment=audio_data[t:t+FFT_SIZE]
                
                # Apply window and pad if needed
                if len(segment)<FFT_SIZE:
                    segment=np.pad(segment,(0,FFT_SIZE-len(segment)),'constant')
                windowed_segment=segment*window
                
                # Using FFT implementation
                fft_result=np.array(compute_fft(windowed_segment))
                freqs=np.fft.fftfreq(FFT_SIZE, 1/params.framerate)[:FFT_SIZE//2]
                magnitude=np.abs(fft_result[:FFT_SIZE//2])
                # Normalize
                magnitude=magnitude/np.max(magnitude)
                # Using original derivative-based peak detection
                peak_freq = find_peak_frequency_by_derivative(freqs,magnitude)
                if peak_freq:
                    print(f"At {t/params.framerate:.1f}s: Peak at {peak_freq/1000:.1f}kHz")
                    plt.plot(freqs/1000, magnitude, color=colors[i],
                           label=f"{t/params.framerate:.1f}s ({peak_freq/1000:.1f}kHz)")
                else:
                    print(f"At {t/params.framerate:.1f}s: No clear peak found")
            
            plt.title("Normalized FFT Magnitude Spectrum")
            plt.xlabel("Frequency (kHz)")
            plt.ylabel("Normalized Magnitude")
            plt.legend()
            plt.xlim(0,20)
            plt.grid(True,alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"Error:{str(e)}")

if __name__=="__main__":
    filename="audiosamplereverse.wav"
    analyze_sweep(filename)


# Resources
# https://builtin.com/articles/fast-fourier-transform
# https://www.dsprelated.com/showarticle/1712.php
# https://www.sweetwater.com/sweetcare/articles/sample-rate/
# Audio Bit Depth and Sample Rate Explained - YouTube  
# https://docs.scipy.org/doc/scipy/index.html