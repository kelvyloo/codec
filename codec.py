
# coding: utf-8

# # ECES 434 Intro to Applied DSP
# ## Final Project

# In[153]:


import librosa
import numpy as np
import sys
import pickle
import time
import IPython.display as ipd


# ---
# ## Import Audio File
# *Load in an audio file to test with codec.*

# In[154]:


filename = "Voices/female3.wav"
x, sr = librosa.load(filename)


# ---
# ## Encode/Decode
# 
# *Define your encode and decode functions.*

# In[155]:


# encode function here
def encode(x, sr=22050, hop=512, n=30):
    
    D = librosa.stft(x).T
    
    a = np.zeros((D.shape[0], 3, n))
    
    q = 3
    
    index = 0
    
    for dft in D:
    
        A = []
        F = []
        P = []

        length = np.shape(dft)[0]
        magSpec = np.abs(dft)
        phase = np.angle(dft)
        freq = (np.arange(length)/length) * (sr//2)

        for peak in range(n):
            i = np.argmax(magSpec)
            h = magSpec[i]
            A.append(h)
            F.append(freq[i])
            P.append(phase[i])
            
            halfWin = int(np.floor(i/(q*2)))
            rampUp = np.linspace(0, h, halfWin, endpoint=False)
            rampDown = np.flip(rampUp, axis=0)
            
            triangle = np.concatenate([rampUp, np.array(h), rampDown], axis=None)
            zeros = np.zeros(length)
            padded = np.concatenate((zeros,triangle,zeros),axis = 0)
            shift = length + halfWin - i
            tri = padded[shift:(shift + length)]
            
            phase = np.array([0 if magSpec[idx] <= tri[idx] else phase[idx] for idx in range(length)])
            magSpec = np.array([0 if (magSpec[idx] <= tri[idx]) else magSpec[idx] for idx in range(length)])

        a[index][0] = A
        a[index][1] = F
        a[index][2] = P

        index += 1
    
    return a


# In[156]:


# decode function here
def decode(x, sr=22050, hop_size=512, num_freq=30):
    y_out = np.zeros((1, 0))
    dur = hop_size/sr
    
    for frame in x:
        amp = frame[0]
        freq = frame[1]
        phase = frame[2]
        
        t = np.linspace(0, dur, hop_size)
        
        freq.shape = (num_freq, 1)
        amp.shape = (1, num_freq)
        t.shape = (1, len(t))
        phase.shape = (num_freq, 1)
        
        cos_matrix = np.cos(2 * np.pi * freq * t + phase)
        cos_sum = np.dot(amp, cos_matrix)
        
        y_out = np.concatenate((y_out, cos_sum), axis=1)
        
    return y_out


# ---
# ## Runtime
# *Encode and decode the audio. Time the processes.*

# In[157]:


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

# In[158]:


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

# In[159]:


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

# In[160]:


print("Total elapsed time for codec:", total_time)
print("\tElapsed time for encode:", encode_elapse_t)
print("\tElapsed time for decode:", decode_elapse_t, "\n")
print("Compression Ratio:", compress_ratio, "\n")
print("Signal-to-Noise Ratio (dB):", snr)


# In[161]:


# Original
ipd.Audio(x, rate = sr)


# In[162]:


# Encoded/Decoded
ipd.Audio(x_decoded, rate = sr)

