import librosa
import numpy as np

"""
This is where you extract the audio features and consider the way of creating generators.

needed attributes:
    self.sr
    self.log
    self.resolution_range
    self.feature_type
    self.window_size,
    self.num_mels,
    self.hop_length
"""
class AudioFeatureExtractor():
    def __init__(
        self,
        **kwargs
        ):

        super().__init__()

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

        feature = (feature - feature.min()) / (feature.max() - feature.min())

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
