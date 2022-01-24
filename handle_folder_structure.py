"""
This file handles the audio data, make a folder with a structure suitable for AudioFeatureGenerator,
i.e, directory/label1, label2, label3,...
"""

from move import create_dir
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

def handle_meta(directory):
    meta = pd.read_csv(directory)
    labels = meta[["label"]]
    meta[["label"]] = LabelEncoder().fit_transform(meta[["label"]].values.ravel())
    meta_data = meta[["label", "fname"]].values
    return meta_data



if __name__ == '__main__':
    meta_dir = "./kaggle_audio_tagging/meta"
    train_meta = handle_meta(meta_dir + "/train.csv")
    test_meta = pd.read_csv(meta_dir + "/test.csv")
    test_meta = test_meta.fname.values[...,None]
    train_meta, valid_meta = train_test_split(train_meta, random_state = 7, test_size = 0.2)
    labels = train_meta[:,0]

    try:
        create_dir(labels)
    except:
        print("data dir already existed")

    def move_file(meta, src_dir, dst_dir, use_label = True):
        for row in meta:
            if use_label:
                fname, label = row[1], row[0]
            else:
                fname = row[0]
            src_file = src_dir + "/" + fname
            if use_label:
                dst_file = dst_dir + "/" + str(label) + "/" + fname
            else:
                dst_file = dst_dir + "/" + fname

            os.replace(src_file, dst_file)


    move_file(train_meta, "./kaggle_audio_tagging/train", "./data/train")
    move_file(valid_meta, "./kaggle_audio_tagging/train", "./data/valid")
    move_file(test_meta, "./kaggle_audio_tagging/test", "./data/test", False)
