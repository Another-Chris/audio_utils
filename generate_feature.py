import librosa
import cv2

import pandas as pd
import numpy as np

from tqdm import tqdm
from shared import Config, get_segment, get_melspectrogram, get_segfeatures

data_dir = f"../input/speech_bundle"
datatype = "cnceleb"

config = Config(
    sr = 16000,
    n_mels = 60,
    win_length = 25e-3,
    hop_length = 10e-3,
    fmin = 20,
    fmax = None,
    seed = 111,
    input_shape = (60,128)
)


frac = 1
train_meta = pd.read_csv(f"{data_dir}/{datatype}/meta/train.csv")

# get split train test
train_sample = train_meta[~train_meta.fdir.isin(test_utterances.fdir)]
train_sample = train_sample.sample(frac = frac, random_state = config.seed, replace = False)


def store_features(meta, train_or_test):
    features = None
    labels = []
    total_utterances = meta.shape[0]

    print(f"[+] collecting {total_utterances} utterances")
    for i in tqdm(range(total_utterances)):

        xdir = f"./features/{datatype}/{train_or_test}/X_{i}.npz"
        ydir = f"./features/{datatype}/{train_or_test}/y_{i}.npz"

        sample = meta.iloc[i]
        fdir = sample.fdir
        label = sample.label

        segfeatures = get_segfeatures(fdir, config)
        nlabels = segfeatures.shape[0]
        labels += nlabels * [label]

        if features is None:
            features = segfeatures
        else:
            features = np.concatenate([features, segfeatures], axis = 0)

        if (i!=0) and (i%1000 ==0):
            labels = np.array(labels)
            np.savez(xdir, x=features)
            np.savez(ydir, x=labels)
            features = None
            labels = []

    labels = np.array(labels)
    np.savez(xdir, x=features)
    np.savez(ydir, x=labels)

print("store train samples")
store_features(train_sample, "train")

print("store test samples")
store_features(test_sample, "test")