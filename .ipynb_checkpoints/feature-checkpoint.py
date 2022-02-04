import numpy as np
import uuid
import os
import librosa
from scipy.io import wavfile

from .utils import standard_norm, fix_sample_length




class FeatureExtractor:
    def __init__(
        self,
        window_size = 1024,
        hop_length = 512,
        log = True,
        sr = None
        ):

        self.window_size = window_size
        self.hop_length = hop_length
        self.log = log
        self.sr = sr

    def load_sr(self, sr):
        self.sr = sr


    def __extract_melspectrogram(self, utterance, num_mels, resolution):

        if num_mels is None:
            raise ValueError("please specify the number of mels")

        feature = librosa.feature.melspectrogram(
            utterance,
            sr = self.sr,
            n_fft = int(self.window_size * resolution),
            n_mels = num_mels,
            hop_length = int(self.hop_length * resolution),
        )
        return feature

    def __extract_mfccs(self, utterance, num_mfccs, resolution, delta):
        if num_mfccs is None:
            raise ValueError("please specify the number of mfccs")

        mfcc =  librosa.feature.mfcc(
            utterance,
            sr = self.sr,
            n_fft = int(self.window_size * resolution),
            n_mfcc = num_mfccs,
            hop_length = int(self.hop_length * resolution),
        )

        if delta == 0:
            feature = mfcc

        elif delta == 1:
            delta1 = librosa.feature.delta(mfcc, order = 1)
            feature = np.concatenate([mfcc,delta1], axis = 0)

        elif delta == 2:
            delta1 = librosa.feature.delta(mfcc, order = 1)
            delta2 = librosa.feature.delta(mfcc, order = 2)
            feature = np.concatenate([mfcc,delta1,delta2], axis = 0)

        return feature

    def extract_feature_from_utterance(
            self,
            utterance,
            resolution,
            feature_type,
            num_mels,
            num_mfccs,
            delta
            ):

        if (feature_type == "mel_spectrogram") or (feature_type == "mspc"):
            feature = self.__extract_melspectrogram(
                utterance,
                num_mels,
                resolution
            )

        elif feature_type == "mfcc":
            feature = self.__extract_mfccs(
                utterance,
                num_mfccs,
                resolution,
                delta
            )

        if self.log:
            feature = librosa.power_to_db(feature)

        feature = standard_norm(feature, axis = 0)
        return feature


    def extract_feature_with_resolutions(
        self,
        utterance,
        feature_type,
        resolution_range = [1],
        num_mels,
        num_mfccs,
        ):
        features = []

        for resolution in resolution_range:
            features.append(
            self.extract_feature_from_utterance(utterance,resolution,feature_type)
            )

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
    def __init__(self, fix_length = 3, random_seg = True, **kwargs):
        super().__init__(**kwargs)

        self.fix_length = fix_length
        self.random_seg = random_seg

    def generate_from_fdirs(self, batch_dirs):
        batch = []
        for fdir in batch_dirs:
            sr, utterance = wavfile.read(fdir)
            feature = self.extract_feature_from_utterance(utterance)
            feature = normalization(feature, axis = 0)
        return np.stack(batch, axis = 0)




class AudioFeatureStorer(FeatureExtractor):
    def __init__(self, fix_length = 3, random_seg = True, **kwargs):
        super().__init__(**kwargs)

        self.fix_length = fix_length
        self.random_seg = random_seg

    def extract_from_fdir(self, fdir):
        sr, utterance = wavfile.read(fdir)
        if self.sr is None:
            self.load_sr(sr)

        if self.fix_length > 0:
            utterance = fix_sample_length(utterance, self.sr * self.fix_length, self.random_seg)
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
