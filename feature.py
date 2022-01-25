import numpy as np
import uuid
import os
import librosa
from scipy.io import wavfile

from utils import standard_norm, fix_sample_length


class FeatureExtractor:
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


class AudioFeatureStorer(FeatureExtractor):
    def __init__(self, fix_length, **kwargs):
        super().__init__(**kwargs)

        self.fix_length = fix_length

    def extract_from_fdir(self, fdir):
        sr, utterance = wavfile.read(fdir)
        if self.sr is None:
            self.load_sr(sr)

        if self.fix_length:
            utterance = fix_sample_length(utterance, self.sr * 3)
        utterance = standard_norm(utterance)
        return self.extract_feature_with_resolutions(utterance)

    def store_one_batch(self,batch_fnames,batch_labels, fhandle):
        batch_features = []
        for fdir in batch_fnames:
            feature = self.extract_from_fdir(fdir)
            batch_features.append(feature)

        batch_features = np.stack(batch_features, axis = 0)
        batch_labels = np.stack(batch_labels, axis = 0)
        np.savez_compressed(fhandle, batch_features,batch_labels)

    def store_feature(self, fdirs, labels, dst_dir, batch_size = 32):
        total_files = int(len(fdirs) / batch_size) + 1
        print(f"start to store, overall {total_files} batches.")

        if labels is None:
            labels = np.zeros(len(fdirs))
            
        if labels.dtype != "int":
            labels = labels.astype("int")

        for i in range(0, len(fdirs), batch_size):
            print(f"storing batch {int(i / 32)}")
            batch_features = []
            batch_fnames = fdirs[i:i+batch_size]

            dst_fname = str(uuid.uuid1()) + ".npz"
            with open(dst_dir + "/" + dst_fname, "wb") as fhandle:
                self.store_one_batch(batch_fnames, labels[i:i+batch_size], fhandle)
