import numpy as np
import os

from audio_tuning_validation import compute_mfcc, gmm_training, load_norm


if __name__ == "__main__":

    dirs = ["target_new", "non_target_new"]
    dirs_aug = ["target_new_aug", "non_target_new_aug"]

    tar_train_set = []
    nontar_train_set = []

    for filename in os.listdir(dirs[0]):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(dirs[0], filename)
        data = load_norm(filepath)
        tar_train_set.append(data)


    for filename in os.listdir(dirs_aug[0]):
        if not filename.endswith(".wav"):
            continue
        
        filepath = os.path.join(dirs_aug[0], filename)
        data = load_norm(filepath)
        tar_train_set.append(data)
        

    for filename in os.listdir(dirs[1]):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(dirs[1], filename)
        data = load_norm(filepath)
        nontar_train_set.append(data)


    for filename in os.listdir(dirs_aug[1]):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(dirs_aug[1], filename)
        data = load_norm(filepath)
        nontar_train_set.append(data)


    # convert signals to mfccs
    tar_train_set = compute_mfcc(tar_train_set)
    nontar_train_set = compute_mfcc(nontar_train_set)

    Ws_t, MUs_t, COVs_t = gmm_training(tar_train_set, 6, 15)

    Ws_nt, MUs_nt, COVs_nt = gmm_training(nontar_train_set, 18, 25)


    if not os.path.exists("audio_gmm_model"):
        os.makedirs("audio_gmm_model")

    np.save('audio_gmm_model/Ws_t.npy', Ws_t)
    np.save('audio_gmm_model/MUs_t.npy', MUs_t)
    np.save('audio_gmm_model/COVs_t.npy', COVs_t)
    np.save('audio_gmm_model/Ws_nt.npy', Ws_nt)
    np.save('audio_gmm_model/MUs_nt.npy', MUs_nt)
    np.save('audio_gmm_model/COVs_nt.npy', COVs_nt)
    