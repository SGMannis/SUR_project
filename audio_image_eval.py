import os
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal Score Fusion (Audio + Face)")
    
    parser.add_argument(
        "--audio_result", 
        type=str, 
        default="audio_gmm/audio_gmm_result.txt",
        help="Path to .txt file with audio model results (default: audio_gmm/audio_gmm_result.txt)"
    )
    
    parser.add_argument(
        "--image_result", 
        type=str, 
        default="face_recognition/image_HOG_LBP_SVM.txt",
        help="Path to .txt file with image model results (default: face_recognition/image_HOG_LBP_SVM.txt)"
    )
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    
    audio_result = args.audio_result
    image_result = args.image_result

    audio_thresh_path = 'audio_gmm/audio_gmm_model/threshold.txt'
    image_thresh_path = 'face_recognition/face_model/threshold.txt'

    with open(audio_thresh_path, 'r') as f:
            line = f.readline().split()
            audio_thresh =  np.float64(line[0])
    with open(image_thresh_path, 'r') as f:
            image_thresh =  np.float64(f.readline())

    new_threshold = 0.5 * image_thresh + 0.5 * audio_thresh

    if os.path.exists(audio_result) and os.path.exists(image_result):
        with open(audio_result, "r", encoding="utf-8") as fa:
            with open(image_result, "r", encoding="utf-8") as fi:
                with open("multimodal_result.txt", "w") as fo:

                    for audio_line, image_line in zip(fa, fi):
                        a_line = audio_line.strip().split()
                        i_line = image_line.strip().split()

                        a_filename = a_line[0]
                        i_filename = i_line[0]
                        # we are evaluating matching image and audio
                        assert(a_filename == i_filename)
                        
                        a_score = np.float64(a_line[1])
                        i_score = np.float64(i_line[1])

                        new_score = 0.5 * a_score + 0.5 * i_score

                        result = 1 if new_score > new_threshold else 0

                        fo.write(f'{a_filename} {new_score} {result}\n')
    else:
        print("Result files not found")
