import tensorflow as tf
from preprocessing import \
    AudioFeatureExtractor,standard_norm,fix_sample_length
import os
import matplotlib.pyplot as plt
import cv2
from scipy.io import wavfile
import numpy as np
import uuid
from move import create_folder_dir


target = "./data/train"
fdirs = os.listdir(target)
sample = target + "/" + fdirs[1]
arr = np.load(sample)
arr["arr_0"].shape
arr["arr_1"].shape
## this could be in the class AudioFeatureGenerator
## because we can reuse afe
## and store the fname for later reference

afe = AudioFeatureExtractor(
    window_size = 1024,
    hop_length = 512,
    resolution_range = [1,2,4],
    feature_type = "mel_spectrogram"
)


def extract_from_fdir(fdir):
    sr, utterance = wavfile.read(fdir)
    if afe.sr is None:
        afe.load_sr(sr)
    utterance = fix_sample_length(utterance, afe.sr * 3)
    utterance = standard_norm(utterance)
    return afe.extract_feature_with_resolutions(utterance)


def store_one_batch(batch_fnames, batch_labels, fhandle):
    batch_features = []
    for fdir in batch_fnames:
        feature = extract_from_fdir(fdir)
        batch_features.append(feature)

    batch_features = np.stack(batch_features, axis = 0)
    batch_labels = np.stack(batch_labels, axis = 0)
    np.savez_compressed(fhandle, batch_features,batch_labels)


def store_feature(fdirs, labels, dst_dir, batch_size = 32):
    for i in range(0, len(fdirs), batch_size):
        batch_features = []
        batch_fnames = fdirs[i:i+batch_size]

        dst_fname = str(uuid.uuid1()) + ".npz"
        with open(dst_dir + "/" + dst_fname, "wb") as fhandle:
            store_one_batch(batch_fnames, labels[i:i+batch_size], fhandle)

root_dir = "./kaggle_audio_tagging"
train_dir=  "./kaggle_audio_tagging/train"
dst_dir = "./data"
create_folder_dir(dst_dir)

train_fdirs = []
train_labels = []
for label in os.listdir(train_dir):
    for fname in os.listdir(train_dir + "/" + str(label)):
        train_fdirs.append(train_dir + "/" + str(label) + "/" + fname)
        train_labels.append(label)


store_feature(train_fdirs, train_labels, dst_dir + "/train")
## store all the files in
