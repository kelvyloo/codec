
# coding: utf-8

# # ECES 434 Intro to Applied DSP
# ## Final Project: Audio Codec

# In[138]:


import librosa
import numpy as np
import sys
import pickle
import time
import IPython.display as ipd
import io


# ---
# ## Import Audio File
# *Load in an audio file to test with codec.*

# In[139]:


filename = "Voices/female3.wav"
#filename = "taxman.wav"
x, sr = librosa.load(filename)


# ## Encoder/Decoder helper functions

# In[140]:


def mask(x, sr, num_freq):
    
    X = librosa.stft(x).T
    
    masked_sig = np.zeros((X.shape[0], 3, num_freq))
    
    triangle_width = 3
    
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

        masked_sig[index][0] = amp_vec
        masked_sig[index][1] = freq_vec
        masked_sig[index][2] = phase_vec

        index += 1
    
    return masked_sig


# In[141]:


def unmask(x, sr, num_freq):
    
    unmasked_sig = np.zeros((1, 0))
    dur = 512/sr
    
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
        
    return unmasked_sig


# In[142]:


def compress(x):
    
    compressed_sig = io.BytesIO()
    np.savez_compressed(compressed_sig, x)
    
    return compressed_sig


# In[143]:


def decompress(x):
    
    x.seek(0)
    decompressed = np.load(x)['arr_0']
    
    return decompressed


# ---
# ## Encode/Decode
# 
# *Definitions for encode and decode functions.*

# In[144]:


# encode function
def encode(x, sr=22050, num_freq=20):
    
    x_mask = mask(x, sr, num_freq)
    
    x_encoded = compress(x_mask)
    
    return x_encoded


# In[145]:


# decode function
def decode(x, sr=22050, num_freq=20):
    
    decompressed = decompress(x)
    
    x_decoded = unmask(decompressed, sr, num_freq)
    
    return x_decoded


# ---
# ## Runtime
# *Encode and decode the audio. Time the processes.*

# In[146]:


# Encode the audio file and also calculate elapsed time
encode_start_t = time.time()
x_encoded = encode(x)
encode_elapse_t = time.time() - encode_start_t

# Decode encoded audio file and calculate elapsed time
decode_start_t = time.time()
x_decoded = decode(x_encoded)
decode_elapse_t = time.time() - decode_start_t

total_time = decode_elapse_t + encode_elapse_t


# ---
# ## Compression Ratio
# *Compare the sizes of the original and encoded structures.*

# In[147]:


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

# In[148]:


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

# In[149]:


print("Total elapsed time for codec:", total_time)
print("\tElapsed time for encode:", encode_elapse_t)
print("\tElapsed time for decode:", decode_elapse_t, "\n")
print("Compression Ratio:", compress_ratio, "\n")
print("Signal-to-Noise Ratio (dB):", snr)


# In[150]:


# Original
ipd.Audio(x, rate = sr)


# In[151]:


# Encoded/Decoded
ipd.Audio(x_decoded, rate = sr)

