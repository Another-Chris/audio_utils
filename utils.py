"""
purpose:
    trim / pad the utterances with different length
    method: random samling
"""

import numpy as np


def standard_norm(vec):
    mean = vec.mean()
    std = vec.std()
    return (vec - mean) / std

def fix_sample_length(sample, sample_per_utterance, randi):
    sample_len = len(sample)
    if sample_len  > sample_per_utterance:
        randi = sample_len - randi
        sample_seg = sample[randi:randi+sample_per_utterance]
    else:
        pad_width = sample_per_utterance - sample_len
        sample_seg = np.pad(sample,(0,pad_width),"constant")
    return standard_norm(sample_seg)
