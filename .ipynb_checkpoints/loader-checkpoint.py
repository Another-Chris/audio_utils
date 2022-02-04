import numpy as np
import os
from scipy.io import wavfile
from .feature_v2 import FeatureExtractor
from .utils import standard_norm, fix_sample_length

def yield_batch(fdirs):
    while True:
        for fdir in fdirs:
            data = np.load(fdir)
            data,label = data["arr_0"], data["arr_1"]
            yield data,label
            
"""
input: list of fdirs
output: batch of features
"""
def batch_generator(
        fdirs, 
        labels,
        batch_size, 
        feature_type, 
        window_size, 
        hop_length, 
        utterance_length,
        resolution_range,
        **kwargs
):
    features = []
    batch_labels = []
    feconfig = dict(
        window_size = window_size,
        hop_length = hop_length,
        feature_type = feature_type
    )
    if feature_type == "mel_spectrogram":
        feconfig["num_mels"] = kwargs["num_mels"]
    elif feature_type == "mfcc":
        feconfig["num_mfccs"] = kwargs["num_mfccs"]
        feconfig["delta"]  = kwargs["delta"]
        
    fe = FeatureExtractor(**feconfig)
    buffer = {}
    
    while True:
        for i in range(len(fdirs) - batch_size):
            sr, data = wavfile.read(fdirs[i])
            if fe.sr is None:
                fe.load_sr(sr)
                
            if buffer.get(i) is not None:
                yield buffer.get(i)[0], buffer.get(i)[1]
                return
                
            data = data.astype("float")
            data = fix_sample_length(data, utterance_length * sr, random = False)
            feature = fe.extract_feature_with_resolutions(data,resolution_range)
            feature = standard_norm(feature, axis = 0)
            features.append(feature)
            batch_labels.append(labels[i])

            if (i > 0) and ( (i+1)% batch_size == 0):
                features = np.stack(features, axis = 0)
                batch_labels = np.array(batch_labels)
                buffer[i] = (features, batch_labels)
                yield features, batch_labels
                features = []
                batch_labels = []

        
        
        
def extract_feature(
    fdirs,
    feature_type, 
    window_size, 
    hop_length, 
    utterance_length,
    resolution_range,
    **kwargs
):
    feconfig = dict(
        window_size = window_size,
        hop_length = hop_length,
        feature_type = feature_type
    )
    if feature_type == "mel_spectrogram":
        feconfig["num_mels"] = kwargs["num_mels"]
    elif feature_type == "mfcc":
        feconfig["num_mfccs"] = kwargs["num_mfccs"]
        feconfig["delta"]  = kwargs["delta"]
        
    fe = FeatureExtractor(**feconfig)
    features = []
    for i,fdir in enumerate(fdirs):
        print(f"storing features... {i}", end = "\r", flush = True)
        sr, data = wavfile.read(fdirs[i])
        if fe.sr is None:
            fe.load_sr(sr)
        data = data.astype("float")
        data = fix_sample_length(data, utterance_length * sr, random = False)
        feature = fe.extract_feature_with_resolutions(data,resolution_range)
        feature = standard_norm(feature, axis = 0)
        features.append(feature)
    return np.stack(features, axis = 0)
        
        