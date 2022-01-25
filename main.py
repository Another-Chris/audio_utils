import numpy as np
from utils import create_folder_dir
from feature import AudioFeatureStorer
import os

afs = AudioFeatureStorer(
    window_size = 1024,
    hop_length = 512,
    resolution_range = [1,2,4],
    feature_type = "mel_spectrogram",
    fix_length = True
)


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


afs.store_feature(train_fdirs, train_labels, dst_dir + "/train", batch_size=32)
