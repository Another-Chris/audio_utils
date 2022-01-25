"""
purpose:
    trim / pad the utterances with different length
    method: random samling
"""

import numpy as np
import os



def standard_norm(vec):
    mean = vec.mean()
    std = vec.std()
    return (vec - mean) / std

def fix_sample_length(sample, sample_per_utterance):
    sample_len = len(sample)
    if sample_len  > sample_per_utterance:
        sample_seg = sample[0:0+sample_per_utterance]
    else:
        pad_width = sample_per_utterance - sample_len
        sample_seg = np.pad(sample,(0,pad_width),"constant")
    return standard_norm(sample_seg)

def create_dir(labels):
    root = "./data"
    train = f"{root}/train"
    valid = f"{root}/valid"
    test = f"{root}/test"

    os.mkdir(root)
    os.mkdir(train)
    os.mkdir(valid)
    os.mkdir(test)
    for label in set(labels):
        os.mkdir(f"{train}/{label}")
        os.mkdir(f"{valid}/{label}")
        os.mkdir(f"{test}/{label}")


def create_folder_dir(root_name):
    root = f"./{root_name}"
    train = f"{root}/train"
    valid = f"{root}/valid"
    test = f"{root}/test"
    try:
        os.mkdir(root)
        os.mkdir(train)
        os.mkdir(valid)
        os.mkdir(test)
    except:
        print("folder already exists")
