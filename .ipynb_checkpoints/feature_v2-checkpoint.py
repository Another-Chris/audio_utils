import librosa
from .utils import standard_norm
import numpy as np


class FeatureExtractor:
    def __init__(
        self,
        feature_type,
        window_size,
        hop_length,
        resolution_range = [1],
        num_mels = None,
        num_mfccs = None,
        delta = 0,
        log = True,
        sr = None
        ):
        self.feature_type = feature_type
        self.window_size = window_size
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.num_mfccs = num_mfccs
        self.log = log
        self.sr = sr
        self.delta = delta
        self.resolution_range = resolution_range

    def load_sr(self, sr):
        self.sr = sr


    def __extract_melspectrogram(self, utterance, resolution):

        if self.num_mels is None:
            raise ValueError("please specify the number of mels to use")

        feature = librosa.feature.melspectrogram(
            utterance,
            sr = self.sr,
            n_fft = int(self.window_size * resolution),
            n_mels = self.num_mels,
            hop_length = int(self.hop_length * resolution),
        )
        return feature

    def __extract_mfccs(self, utterance, resolution):

        if self.num_mfccs is None:
            raise ValueError("please specify the number of mfccs to use")

        mfcc =  librosa.feature.mfcc(
            utterance,
            sr = self.sr,
            n_fft = int(self.window_size * resolution),
            n_mfcc = self.num_mfccs,
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

        return feature

    def extract_feature_from_utterance(self,utterance,resolution):

        if self.feature_type is None:
            raise ValueError("please specify a feature type to use")

        if (self.feature_type == "mel_spectrogram") or (self.feature_type == "mspc"):

            feature = self.__extract_melspectrogram(
                utterance,
                resolution
            )

        elif self.feature_type == "mfcc":
            feature = self.__extract_mfccs(
                utterance,
                resolution,
            )

        if self.log:
            feature = librosa.power_to_db(feature)

        return feature


    def extract_feature_with_resolutions(self,utterance):
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
