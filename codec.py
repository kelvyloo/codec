
# coding: utf-8

# # Week 4 Recitation
# ## Final Project Overview
# 
# *This week, we will go over the final project prompt and create a test script together.*
# 
# ---

# In[1]:


import librosa
import numpy as np
import sys
import pickle
import time
import IPython.display as ipd


# ---
# ## Import Audio File
# *Load in an audio file to test with codec.*

# In[2]:


filename = "taxman.wav"
x, sr = librosa.load(filename)


# ---
# ## Encode/Decode
# 
# *Define your encode and decode functions.*

# In[3]:


# your encode function here
def encode(a):
    a = librosa.resample(a, sr, sr/2)
    return a


# In[4]:


# your decode function here
def decode(a):
    a = librosa.resample(a, sr/2, sr)
    return a


# ---
# ## Runtime
# *Encode and decode the audio. Time the processes.*

# In[5]:


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

# In[6]:


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

# In[7]:


def signal_to_noise(original, decoded):
    # force the signals to be same dimensions
    length_diff = len(original) - len(decoded)
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

# In[8]:


print("Total elapsed time for codec:", total_time)
print("\tElapsed time for encode:", encode_elapse_t)
print("\tElapsed time for decode:", decode_elapse_t, "\n")
print("Compression Ratio:", compress_ratio, "\n")
print("Signal-to-Noise Ratio (dB):", snr)


# In[9]:


# Original
ipd.Audio(x, rate = sr)


# In[10]:


# Encoded/Decoded
ipd.Audio(x_decoded, rate = sr)


# ---
# 
# ## Small Task
# 
# Change the encode function above so that it simply downsamples the signal by a factor of 2. Change the decode function so that it upsamples it back to its original sampling rate. Compute the **compression ratio**, **runtime**, and **SNR** for this compression. Listen to the decoded signal and compare it to the original.
# 
# ***Hint:*** You may considering using the librosa.resample() function.
