from scipy.io import wavfile
import os
import random
import numpy as np
import librosa

def standard_norm(vec):
    mean = vec.mean()
    std = vec.std()
    return (vec - mean) / std

def create_fdirs(directory):
    total_labels = os.listdir(directory)
    fdirs = []
    for label in total_labels:
        fnames_for_currlabel = os.listdir(directory + "/" + label)
        for fname in fnames_for_currlabel:
            fdirs.append((directory + "/" + label + "/" + fname, label))
    random.shuffle(fdirs)
    return [el[0] for el in fdirs], [el[1] for el in fdirs]


def fix_sample_length(sample, sample_per_utterance):
    sample_len = len(sample)
    if sample_len  > sample_per_utterance:
        sample_seg = sample[0:sample_per_utterance]
    else:
        pad_width = sample_per_utterance - sample_len
        sample_seg = np.pad(sample,(0,pad_width),"constant")
    return standard_norm(sample_seg)

class AudioFeatureExtractor():
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

        self.feature_type = feature_type
        self.resolution_range = resolution_range
        self.window_size = window_size
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.num_mfccs = num_mfccs
        self.delta = delta
        self.log = log
        self.sr = None

    def load_sr(self, sr):
        self.sr = sr

    def extract_feature_from_utterance(self, utterance, resolution):

        if (self.feature_type == "mel_spectrogram") or (self.feature_type == "mspc"):
            feature =  librosa.feature.melspectrogram(
                utterance,
                sr = self.sr,
                n_fft = int(self.window_size * resolution),
                n_mels = self.num_mels,
                hop_length = int(self.hop_length * resolution),
            )


        if self.feature_type == "mfcc":
            mfcc =  librosa.feature.mfcc(
                utterance,
                sr = self.sr,
                n_fft = int(self.window_size * resolution),
                n_mfcc = self.num_mels,
                hop_length = int(self.hop_length * resolution),
            )

            if self.delta == 0:
                feature = mfcc

            elif self.delta == 1:
                delta1 = librosa.feature.delta(mfcc, order = 1)
                feature = np.concatenate([mfcc,delta1], axis = 0)

            elif self.delta == 2:
                delta1 = librosa.feature.delta(mfcc, order = 1)
                delta2 = librosa.feature.delta(mfcc, order = 2)
                feature = np.concatenate([mfcc,delta1,delta2], axis = 0)


        if self.log:
            feature = librosa.power_to_db(feature)

        # feature = (feature - feature.min()) / (feature.max() - feature.min())
        feature = standard_norm(feature)
        return feature

    def extract_feature_with_resolutions(self, utterance):
        features = []

        for resolution in self.resolution_range:
            features.append(self.extract_feature_from_utterance(utterance,resolution))

        ## multiple resolution, pad each and stack them
        largest = max(feature.shape[1] for feature in features)
        new_features = []
        for feature in features:
            length = feature.shape[1]
            if length < largest:
                feature = np.pad(feature, ((0,0), (0, largest - length)), "constant")
            new_features.append(feature)
        return np.stack(new_features, axis = -1)




class AudioFeatureGenerator(AudioFeatureExtractor):
    def __init__(
            self, **kwargs

        ):
        super().__init__(**kwargs)

        self.fdirs = None
        self.labels = None
        self.randi = None

    def yield_batch(self, batch_size, sample_per_utterance):
        while True:
            for i in range(0,len(self.fdirs) - batch_size,batch_size):
                features = []
                for batch_idx in range(i,i+batch_size):
                    sample,label = self.fdirs[batch_idx], self.labels[batch_idx]
                    sample_data = wavfile.read(sample)[1].astype(float)
                    sample_data = fix_sample_length(
                            sample_data,
                            sample_per_utterance
                        )
                    ## extract features
                    feature = self.extract_feature_with_resolutions(sample_data)
                    features.append(feature)
                yield np.stack(features, axis = 0), np.stack(self.labels[i:i+batch_size]).astype(int)


    def yield_sample(self):
        while True:
            for i in range(len(self.fdirs)):
                sample,label = self.fdirs[i], self.labels[i]
                sample_data = wavfile.read(sample)[1].astype(float)
                ## extract features
                feature = self.extract_feature_with_resolutions(sample_data)
                yield feature, label


    def flow_from_directory(
        self,
        directory,
        utterance_length = 3,
        batch_size = 32,
        fix_length = False
        ):
        if self.fdirs is None:
            self.fdirs, self.labels = create_fdirs(directory)

        self.sr = wavfile.read(self.fdirs[0])[0]
        sample_per_utterance = utterance_length * self.sr

        if not fix_length:
            return self.yield_sample()
        else:
            return self.yield_batch(batch_size,sample_per_utterance)
