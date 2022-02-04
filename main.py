from feature_v2 import FeatureExtractor
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

window_size = 1024
hop_length = window_size // 2.5


root_dir = "../kaggle_audio_tagging"
train_dir = root_dir + "/" + "train"
fdir = train_dir + "/" + os.listdir(train_dir)[3]

sr, data = wavfile.read(fdir)

fe = FeatureExtractor(
    feature_type = "mel_spectrogram",
    window_size = window_size,
    hop_length = hop_length,
    sr = sr,
    num_mels = 128,
)

data = data.astype("float")
feature = fe.extract_feature_from_utterance(data, 1)
plt.imshow(feature)
plt.show()
