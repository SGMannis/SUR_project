import numpy as np
from scipy.io import wavfile
from sklearn.metrics import roc_curve
from scipy.cluster.vq import kmeans
import os

from ikrlib_stolen import train_gmm, logpdf_gmm, mfcc

# sample rate
SR = 16000

UNIT16MAX = 32767


def find_eer_threshold(target_scores, nontarget_scores):

    all_scores = np.concatenate([target_scores, nontarget_scores])
    all_labels = np.concatenate([np.ones(len(target_scores)), np.zeros(len(nontarget_scores))])

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    frr = 1 - tpr

    # EER => difference between FAR (fpr) a FRR is the smallest
    eer_index = np.argmin(np.abs(fpr - frr))
    eer_threshold = thresholds[eer_index]
    eer_value = fpr[eer_index] 

    return eer_threshold, eer_value


def compute_mfcc(data_list):
    mfcc_list = []
    for data in data_list:
        coeffs = mfcc(data, 400, 240, 512, SR, 23, 13)
        mfcc_list.append(coeffs)
    return mfcc_list


def gmm_training(data, comps, iters):

    data = np.vstack(data)

    np.random.seed(42)                          # so you can reproduce my results ;)

    MUs, _ = kmeans(data, comps)                # use k-means to estimate MUs
    COVs = [np.var(data, axis=0)] * comps       # diagonal COVs
    Ws = np.ones(comps) / comps                 # uniform weights

    for jj in range(iters):
        Ws, MUs, COVs, TTL_t = train_gmm(data, Ws, MUs, COVs)
        # print('Iteration: %d Total log-likelihood: %f' % (jj, TTL_t))

    return Ws, MUs, COVs


def load_norm(filepath):
    rate, data = wavfile.read(filepath) # read as int16
    assert(rate == SR)
    data = data / UNIT16MAX # normalize
    return data



if __name__ == "__main__":

    dirs = ["target_new", "non_target_new"]
    dirs_aug = ["target_new_aug", "non_target_new_aug"]
    val_tar = ["01", "02", "03"]
    val_nontar = [['f401', 'f402', 'f403', 'm414', 'm416', 'm417', 'm419'], ['f404', 'f405', 'f406', 'm420', 'm421', 'm422'], ['f407', 'f408', 'f409', 'm423', 'm424', 'm425']]

    scores_tar = np.array([])
    scores_nontar = np.array([])

    for sitting, people in zip(val_tar, val_nontar):
        tar_train_set = []
        tar_val_set = []
        nontar_train_set = []
        nontar_val_set = []

        for filename in os.listdir(dirs[0]):
            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(dirs[0], filename)
            data = load_norm(filepath)
            if filename.split("_")[1] == sitting:
                tar_val_set.append(data)
            else:
                tar_train_set.append(data)
        

        for filename in os.listdir(dirs_aug[0]):
            if not filename.endswith(".wav"):
                continue
            
            filepath = os.path.join(dirs_aug[0], filename)
            data = load_norm(filepath)
            if filename.split("_")[1] != sitting:
                tar_train_set.append(data)
            
            

        for filename in os.listdir(dirs[1]):
            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(dirs[1], filename)
            data = load_norm(filepath)
            if filename.split("_")[0] in people:
                nontar_val_set.append(data)
            else:
                nontar_train_set.append(data)


        for filename in os.listdir(dirs_aug[1]):
            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(dirs_aug[1], filename)
            data = load_norm(filepath)
            if not (filename.split("_")[0] in people):
                nontar_train_set.append(data)


        # convert signals to mfccs
        tar_train_set = compute_mfcc(tar_train_set)
        tar_val_set = compute_mfcc(tar_val_set)
        nontar_train_set = compute_mfcc(nontar_train_set)
        nontar_val_set = compute_mfcc(nontar_val_set)

        Ws_t, MUs_t, COVs_t = gmm_training(tar_train_set, 6, 15)

        Ws_nt, MUs_nt, COVs_nt = gmm_training(nontar_train_set, 18, 25)

        score = np.array([])
        for tst in tar_val_set:
            ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
            ll_n = logpdf_gmm(tst, Ws_nt, MUs_nt, COVs_nt)
            # priors are 0.5, so it cancels out
            score = np.append(score, np.mean(ll_t) - np.mean(ll_n))
        scores_tar = np.append(scores_tar, score)

        score = np.array([])
        for tst in nontar_val_set:
            ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
            ll_n = logpdf_gmm(tst, Ws_nt, MUs_nt, COVs_nt)
            # priors are 0.5, so it cancels out
            score = np.append(score, np.mean(ll_t) - np.mean(ll_n))
        scores_nontar = np.append(scores_nontar, score)


    MIN_SCORE = min(np.min(scores_tar), np.min(scores_nontar))
    MAX_SCORE = max(np.max(scores_tar), np.max(scores_nontar))

    # Min-Max scaling
    scores_tar_norm = (scores_tar - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
    scores_nontar_norm = (scores_nontar - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)

    threshold_norm, _ = find_eer_threshold(scores_tar_norm, scores_nontar_norm)
    
    print(f'Norm EER Threshold (0-1): {threshold_norm}')
    print(f"Fraction of correctly recognized targets: {np.mean(scores_tar_norm > threshold_norm)}")
    print(f"Fraction of correctly recognized non-targets: {np.mean(scores_nontar_norm < threshold_norm)}")

    if not os.path.exists("audio_gmm_model"):
        os.makedirs("audio_gmm_model")
        
    # save threshold and min and max for normalization
    with open('audio_gmm_model/threshold.txt', 'w') as f:
        f.write(f"{threshold_norm} ")
        f.write(f"{MIN_SCORE} ")
        f.write(f"{MAX_SCORE}")

