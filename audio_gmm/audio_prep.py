import numpy as np
from scipy.io import wavfile
import noisereduce as nr
import os
import librosa

# sample rate
SR = 16000

UNIT16MAX = 32767


def save_wav(filepath, data, sr=SR):
    # save as int16
    data_int16 = np.int16(data * UNIT16MAX)
    
    wavfile.write(filepath, sr, data_int16)


def load_and_cut(path):
    rate, data = wavfile.read(path) # read as int16
    assert(rate == SR)

    data = data.astype(np.float32)[SR:-SR] # cut a second at the beginning and the end
    data = data / UNIT16MAX # normalize

    data_denoised = nr.reduce_noise(y=data, sr=SR) # denoise data
    data_split = librosa.effects.split(data_denoised, top_db=35) # extract the non-silent parts (idices)

    # concatenate non-silent parts of the original recording based on the silent parts of the denoised one
    data = np.concatenate([data[start:end] for start, end in data_split]) 

    return data



if __name__ == "__main__":

    dir_dict = {
        "target_new" : ["target_train", "target_dev"],
        "non_target_new" : ["non_target_train", "non_target_dev"]
    }

    for tar_nontar, dirs in dir_dict.items():

        if not os.path.exists(tar_nontar):
            os.makedirs(tar_nontar)

        for dir in dirs:
            for filename in os.listdir(dir):
                if not filename.endswith(".wav"):
                    continue

                filepath = os.path.join(dir, filename)
                data = load_and_cut(filepath)
                new_filepath = os.path.join(tar_nontar, filename)

                save_wav(new_filepath, data)
