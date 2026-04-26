import numpy as np
from scipy.io import wavfile
import noisereduce as nr
import os
import random

from audio_prep import save_wav

# sample rate
SR = 16000

UNIT16MAX = 32767


##############################
# augmentation functions
##############################
def echo(data):
    delay = int(SR * random.uniform(0.02, 0.15)) # delay from 20 to 150 ms
    echo = np.roll(data, delay) * random.uniform(0.15, 0.45) # echo volume from 15 to 45%
    echo[:delay] = 0 # no echo at the beginning
    data_echo = data + echo
    return data_echo

def add_noise(data):
    noise_var = random.uniform(0.005, 0.025) * np.max(np.abs(data))
    data += np.random.normal(0, noise_var, data.shape)
    return data

def clip(data):
    limit = np.max(np.abs(data)) * random.uniform(0.6, 0.8)
    data = np.clip(data, -limit, limit)
    return data
   
def lower_vol(data):
    data = data * random.uniform(0.3, 0.8) # lower volume (30 to 80)
    return data 

def mask(data):
    # cut 50-150ms
    cut_len = int(SR * random.uniform(0.05, 0.15)) 
    start_index = np.random.randint(0, len(data) - cut_len)
    # set 5-10% of samples to zero - starting on `start_index`
    data[start_index : start_index + cut_len] = 0.0
    return data

def inverse(data):
    return data * (-1)


def augment_audio(data_original):
    changed = False
    data = data_original.copy()
    while not changed:
        if random.random() < 0.3:
            data = echo(data)
            changed = True
    
        if random.random() < 0.5:
            data = add_noise(data)
            changed = True

        if random.random() < 0.3:
            data = clip(data)
            changed = True

        if random.random() < 0.3:
            data = lower_vol(data)
            changed = True

        if random.random() < 0.4:
            data = mask(data)
            changed = True
            
        if random.random() < 0.5:
            data = inverse(data)
            changed = True
        
    return data

##############################
##############################


if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    if not os.path.exists("target_new_aug"):
        os.makedirs("target_new_aug")

    if not os.path.exists("non_target_new_aug"):
        os.makedirs("non_target_new_aug")

    for filename in os.listdir("target_new"):
            filepath = os.path.join("target_new", filename)
            rate, data = wavfile.read(filepath) # read as int16
            assert(rate == SR)
            data = data / UNIT16MAX # normalize

            clean_data = nr.reduce_noise(y=data, sr=SR)

            filepath = os.path.join("target_new_aug", filename)
            new_filepath = filepath.split(".")[0] + "0.wav"

            save_wav(new_filepath, clean_data, sr=SR)
            
            for i in range(1,13):
                data_aug = augment_audio(clean_data)
                new_filepath = filepath.split(".")[0] + str(i) + ".wav"
                save_wav(new_filepath, data_aug, sr=SR)


    for filename in os.listdir("non_target_new"):
            filepath = os.path.join("non_target_new", filename)
            rate, data = wavfile.read(filepath) # read as int16
            assert(rate == SR)
            data = data / 32767 # normalize

            clean_data = nr.reduce_noise(y=data, sr=SR)

            filepath = os.path.join("non_target_new_aug", filename)
            new_filepath = filepath.split(".")[0] + "0.wav"

            save_wav(new_filepath, clean_data, sr=SR)
            
            data_aug = augment_audio(clean_data)
            new_filepath = filepath.split(".")[0] + "1.wav"
            save_wav(new_filepath, data_aug, sr=SR)

