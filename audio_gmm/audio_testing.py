import numpy as np
import os
import argparse

from audio_prep import load_and_cut
from ikrlib_stolen import logpdf_gmm, mfcc

# sample rate
SR = 16000


def parse_arguments():
    parser = argparse.ArgumentParser(description="Testing .wav recordings")

    parser.add_argument(
        "testing_data_dir", 
        type=str,
        help="directory with testing data"
    )

    return parser.parse_args()


def norm_score(score, min_val, max_val):

    score_clipped = np.clip(score, min_val, max_val)    
    # min max scaler
    return (score_clipped - min_val) / (max_val - min_val)


if __name__ == "__main__":

    args = parse_arguments()
    dir = args.testing_data_dir
    if not os.path.isdir(dir):
        print(f"Dir '{dir}' not found")
        exit()

    # target model
    Ws_t = np.load("audio_gmm_model/Ws_t.npy")
    MUs_t = np.load("audio_gmm_model/MUs_t.npy")
    COVs_t = np.load("audio_gmm_model/COVs_t.npy")

    # nontarget model
    Ws_nt = np.load("audio_gmm_model/Ws_nt.npy")
    MUs_nt = np.load("audio_gmm_model/MUs_nt.npy")
    COVs_nt = np.load("audio_gmm_model/COVs_nt.npy")

    # model threshold
    with open('audio_gmm_model/threshold.txt', 'r') as f:
        line = f.readline().split()
        threshold = np.float64(line[0])
        MIN_SCORE = np.float64(line[1])
        MAX_SCORE = np.float64(line[2])


    with open("audio_gmm_results.txt", "w") as f:
        for filename in sorted(os.listdir(dir)):
            if not filename.endswith(".wav"):
                continue

            seg = filename.split(".")[0]
            filepath = os.path.join(dir, filename)
            data = load_and_cut(filepath)

            # should never happen but if it does, this prevents it from crash
            if data.shape[0] == 0:
                continue

            data_features = mfcc(data, 400, 240, 512, SR, 23, 13)

            ll_t = logpdf_gmm(data_features, Ws_t, MUs_t, COVs_t)
            ll_n = logpdf_gmm(data_features, Ws_nt, MUs_nt, COVs_nt)
            score = np.mean(ll_t) - np.mean(ll_n)

            score = norm_score(score, MIN_SCORE, MAX_SCORE)

            result = 1 if score > threshold else 0

            f.write(f'{seg} {score} {result}\n')
