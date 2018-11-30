
# coding: utf-8

# # ECES 434 Intro to Applied DSP
# ## Final Project: Audio Codec

# In[16]:


import librosa
import numpy as np
import sys
import pickle
import time
import IPython.display as ipd
import io
from scipy import signal
import matplotlib.pyplot as plt


# ---
# ## Import Audio File
# *Load in an audio file to test with codec.*

# In[17]:


#filename = "Voices/female3.wav"
#filename = "Voices/female2.wav"
#filename = "Voices/female1.wav"
#filename = "Voices/male2.wav"
#filename = "Voices/male1.wav"
filename = "taxman.wav"
#filename = "jumpman.wav"
x, sr = librosa.load(filename, sr=44100)


# ## Encoder/Decoder helper functions

# In[18]:


def mask(x, sr, num_freq, hop_size):
    
    X = librosa.stft(x, hop_length=hop_size).T
    
    masked_sig = np.zeros((X.shape[0], 3, num_freq))
    
    triangle_width = 0.75
    
    index = 0
    
    for dft in X:
        
        amp_vec = []
        freq_vec = []
        phase_vec = []

        length = np.shape(dft)[0]
        mag = np.abs(dft)
        phase = np.angle(dft)
        freq = (np.arange(length)/length) * (sr//2)
        
        for peak in range(num_freq):
            
            i = np.argmax(mag)
            h = mag[i]
            
            amp_vec.append(h)
            freq_vec.append(freq[i])
            phase_vec.append(phase[i])
            
            half_win = int(np.floor(i/(triangle_width*2)))
            ramp_up = np.linspace(0, h, half_win, endpoint=False)
            ramp_down = np.flip(ramp_up, axis=0)
            
            triangle = np.concatenate([ramp_up, np.array(h), ramp_down], axis=None)
            
            zeros = np.zeros(length)
            padded = np.concatenate((zeros,triangle,zeros),axis = 0)
            
            shift = length + half_win - i
            tri = padded[shift:(shift + length)]
            
            mag = np.array([0 if (mag[j] <= tri[j]) else mag[j] for j in range(length)])
        
        if index % 500 == 0:
            print("Encoding...")
        
        masked_sig[index][0] = amp_vec
        masked_sig[index][1] = freq_vec
        masked_sig[index][2] = phase_vec

        index += 1
    
    return masked_sig


# In[19]:


def unmask(x, sr, num_freq, hop_size):
    
    i = 0
    
    unmasked_sig = np.zeros((1, 0))
    dur = hop_size/sr
    
    for frame in x:
        
        amp = frame[0]
        freq = frame[1]
        phase = frame[2]
        
        t = np.linspace(0, dur, dur * sr)
        
        freq.shape = (num_freq, 1)
        amp.shape = (1, num_freq)
        t.shape = (1, len(t))
        phase.shape = (num_freq, 1)
        
        cos_matrix = np.cos(2 * np.pi * freq * t + phase)
        cos_sum = np.dot(amp, cos_matrix)
        
        unmasked_sig = np.concatenate((unmasked_sig, cos_sum), axis=1)
        
        if i % 500 == 0:
            print("Decoding...")
        
        i += 1
        
    return unmasked_sig


# In[20]:


def compress(x):
    
    compressed_sig = io.BytesIO()
    np.savez_compressed(compressed_sig, x)
    
    return compressed_sig


# In[21]:


def decompress(x):
    
    x.seek(0)
    decompressed = np.load(x)['arr_0']
    
    return decompressed


# In[22]:


def freq_filter(x, sr, low, high):
    
    lower_freq = low/sr
    upper_freq = high/sr
    n = 7
    
    if lower_freq == 0 and upper_freq > 0:
        btype = 'lowpass'
        filter_freqs = [upper_freq]
    elif upper_freq == 0 and lower_freq > 0:
        btype = 'highpass'
        filter_freqs = [upper_freq]
    else:
        btype = 'bandpass'
        filter_freqs = [lower_freq, upper_freq]
        
    [b, a] = signal.butter(n, filter_freqs, btype=btype)
    
    filter_x = signal.lfilter(b, a, x)
    
    return filter_x


# ---
# ## Encode/Decode
# 
# *Definitions for encode and decode functions.*

# In[23]:


# encode function
def encode(x, sr=44100, num_freq=15, hop_size=1024):
    
    x_encoded = compress(x)
    x_encoded = decompress(x_encoded)
    
    x_encoded = mask(x_encoded, sr, num_freq, hop_size)

    x_encoded = compress(x_encoded)
    
    return x_encoded


# In[24]:


# decode function
def decode(x, sr=44100, num_freq=15, hop_size=1024):

    x_decoded = decompress(x)
    
    x_decoded = unmask(x_decoded, sr, num_freq, hop_size)
    
    x_decoded = freq_filter(x_decoded, sr, low=0, high=8000)
    
    x_decoded = x_decoded / np.amax(x_decoded)
    
    return x_decoded


# ---
# ## Runtime
# *Encode and decode the audio. Time the processes.*

# In[25]:


# Encode the audio file and also calculate elapsed time
encode_start_t = time.time()
x_encoded = encode(x)
encode_elapse_t = time.time() - encode_start_t

# Decode encoded audio file and calculate elapsed time
decode_start_t = time.time()
x_decoded = decode(x_encoded)
decode_elapse_t = time.time() - decode_start_t

total_time = decode_elapse_t + encode_elapse_t

plt.figure()
plt.plot(x);

plt.figure()
plt.plot(x_decoded.T);

# original
x_fft = np.fft.fft(x)
x_freq = np.fft.fftfreq(x.size, 1/sr)
x_mag = np.abs(x_fft).T
x_phase = np.angle(np.imag(x_fft)/np.real(x_fft))

plt.figure()
plt.plot(x_freq, x_mag);

# decoded
decoded_fft = np.fft.fft(x_decoded)
decoded_freq = np.fft.fftfreq(x_decoded.size, 1/sr)
decoded_mag = np.abs(decoded_fft).T
decoded_phase= np.angle(np.imag(decoded_fft)/np.real(decoded_fft))

plt.figure()
plt.plot(decoded_freq, decoded_mag);


# ---
# ## Compression Ratio
# *Compare the sizes of the original and encoded structures.*

# In[26]:


def compression_ratio(original, encoded):
    # Serialize data to uniformly compare the original and encoded
    orig_str = pickle.dumps(original)
    encode_str = pickle.dumps(encoded)
    
    # Find ratio between orig and encoded signal
    return sys.getsizeof(orig_str)/sys.getsizeof(encode_str)

compress_ratio = compression_ratio(x, x_encoded)


# ---
# ## SNR
# *Compare the original signal content to the decoded version*

# In[27]:


def signal_to_noise(original, decoded):
    decoded = np.transpose(decoded)
    
    # force the signals to be same dimensions
    length_diff = len(original) - decoded.shape[0]
    if length_diff < 0:
        decoded = decoded[:-length_diff]
    elif length_diff > 0:
        decoded = np.append(decoded, np.zeros((length_diff, 1)))

    # compute SNR
    signal = np.power(original, 2)
    noise = np.power((original - decoded), 2)
    
    # check for divide by zeros and other mathematical errors
    signal = np.where(signal == 0, np.finfo(np.float32).eps, signal)
    noise = np.where(noise == 0, np.finfo(np.float32).eps, noise)
    
    return np.mean(10 * np.log10(signal/noise))

snr = signal_to_noise(x, x_decoded)


# ---
# ## Evaluate Codec
# *Print out evalutation of codec. Listen to the results*

# In[28]:


print("Total elapsed time for codec:", total_time)
print("\tElapsed time for encode:", encode_elapse_t)
print("\tElapsed time for decode:", decode_elapse_t, "\n")
print("Compression Ratio:", compress_ratio, "\n")
print("Signal-to-Noise Ratio (dB):", snr)


# In[29]:


# Original
ipd.Audio(x, rate = sr)


# In[30]:


# Encoded/Decoded
ipd.Audio(x_decoded, rate = sr)

