from scipy.io import wavfile
from feature import AudioFeatureExtractor
from utils import standard_norm
import os
import random
import numpy as np



def create_fdirs(directory):
    total_labels = os.listdir(directory)
    fdirs = []
    for label in total_labels:
        fnames_for_currlabel = os.listdir(directory + "/" + label)
        for fname in fnames_for_currlabel:
            fdirs.append((directory + "/" + label + "/" + fname, label))
    # random.shuffle(fdirs)
    return [el[0] for el in fdirs], [el[1] for el in fdirs]




class AudioFeatureGenerator(AudioFeatureExtractor):
    def __init__(
            self,
            feature_type = "mel_spectrogram",
            resolution_range = [1],
            window_size = 1024,
            hop_length = 512,
            num_mels = 128,
            num_mfccs = 13,
            delta = 0,
            log = True,
        ):
        super().__init__()

        self.feature_type = feature_type
        self.resolution_range = resolution_range
        self.window_size = window_size
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.num_mfccs = num_mfccs
        self.delta = delta
        self.log = log

        self.fdirs = None
        self.labels = None
        self.randi = None



    def flow_from_directory(
        self,
        directory,
        utterance_length = 3,
        batch_size = 32
        ):
        if self.fdirs is None:
            self.fdirs, self.labels = create_fdirs(directory)

        self.sr = wavfile.read(self.fdirs[0])[0]
        sample_per_utterance = utterance_length * self.sr

        while True:
            for i in range(len(self.fdirs) - batch_size):
                features = []
                batch = self.fdirs[i:i+batch_size]
                batch_labels = self.labels[i:i+batch_size]
                for sample in batch:
                    sample_data = wavfile.read(sample)[1]

                    ## extract features
                    feature = self.extract_feature_with_resolutions(sample_fixed_len)
                    features.append(feature)
                yield np.stack(features, axis = 0), np.stack(batch_labels, axis = 0).astype(int)
