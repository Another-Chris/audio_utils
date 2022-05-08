import re
import librosa
import librosa.display
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def inspect_mspec(mspec):
    librosa.display.specshow(mspec, cmap = "rocket")
    plt.show()

def inspect_batch(batch):
    idx = np.random.choice(range(batch.shape[0]), 1)[0]
    inspect_mspec(batch[idx, ...])

def norm_minmax_frame(frame):
    fmin = frame.min(axis = (1,2))[..., None, None]
    fmax = frame.max(axis = (1,2))[..., None, None]
    return (frame - fmin) / (fmax - fmin + 1e-5)


def get_secondary(string):
    return re.findall(r"(\w+)", string)



""" configuration """
class Config:
    def __init__(
            self,
            sr=32000,
            win_length = 25e-3,
            hop_length = 10e-3,
            feature_type = "mspec",
            n_fft=2048,
            n_mels = 128,
            fmin = 20,
            fmax = None,
            duration = 1.27,
            seghop = 0.64,
        
            seed = 111,
            fold = 5,
            epochs = 200,
            steps = 512,
            batch_size = 32,
            learning_rate = 1e-4,
            input_shape = (128, 501),
            n_classes = 21
    ):

        # general
        self.sr = sr
        self.win_length = int(win_length * sr)
        self.hop_length = int(hop_length * sr)
        self.feature_type = feature_type

        # features
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.seghop = seghop

        # others
        self.seed = seed
        self.n_fold = fold

        # model
        self.epochs = epochs
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.n_classes = n_classes



""" utils """
def random_pad(signal, seglen):
    pad_len = int(seglen) - len(signal)

    if pad_len < len(signal):
        return np.concatenate([signal, signal[:pad_len]])

    # pad itself repeatly
    while pad_len > len(signal):
        signal = np.concatenate([signal, signal])
        pad_len = int(seglen) - len(signal)

    # trim the unnecessary part
    if len(signal) > seglen:
        signal = signal[:seglen]
    else:
        signal = np.concatenate([signal, signal[:pad_len]])

    return signal

def random_pad_frame(frame, seglen):
    flen = frame.shape[1]
    pad_len = seglen - flen

    if pad_len < flen:
        return np.concatenate([frame, frame[:, :pad_len]], axis = 1)

    # pad itself repeatly
    while pad_len > flen:
        frame = np.concatenate([frame, frame], axis = 1)
        flen = frame.shape[1]
        pad_len = seglen - flen

    # trim the unnecessary part

    if flen > seglen:
        frame = frame[:, :seglen]
    else:
        frame = np.concatenate([frame, frame[:, :pad_len]], axis = 1)

    return frame

def random_seg(signal, seglen):
    randi = np.random.randint(0,len(signal) - seglen)
    signal = signal[randi:randi+seglen]
    return signal


def random_seg_frame(spec, seglen):
    std = 0
    t = spec.shape[1]
    while std < 1e-5:
        randi = np.random.randint(0, t - seglen)
        spec = spec[:, randi:randi+seglen]
        std = spec.std()
    return spec

# normalize the features in each analysis window
def norm_std(frame, axis = None):
    mean = frame.mean(axis = axis)
    std = frame.std(axis = axis)
    return (frame - mean) / std

def norm_minmax(frame, axis = None):
    min = frame.min(axis = axis)
    max = frame.max(axis = axis)
    return (frame - min) / (max - min + 1e-5)


def get_segment(signal, config, duration=5, seghop=2.5):
    sr = config.sr
    seglen = int(duration * sr)
    segs = []
    for i in range(0, len(signal), int(seghop * sr)):
        seg = signal[i:i+int(duration*sr)]
        if len(seg) < seglen:
            seg = random_pad(seg, seglen)
        segs.append(seg)
    return segs

def get_segfeatures(fdir, config):
    signal, _ = librosa.load(fdir, sr = config.sr)
    segments = get_segment(signal, config, duration = config.duration, seghop = config.seghop)
    segfeatures = []
    for segment in segments:
        feature = get_melspectrogram(segment, config)
        if feature.std() == 0:
            continue
        feature = cv2.normalize(feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        segfeatures.append(feature)
    segfeatures = np.stack(segfeatures, axis = 0)
    return segfeatures


""" preprocess """
def envelope(waveform, wsize=int(32000 * 25e-3 * 8), percentile = 75):
    waveform = (waveform - waveform.mean()) / waveform.std()
    waveform = pd.Series(waveform).apply(np.abs)
    threshold = np.percentile(waveform, percentile)
    y_mean = waveform.rolling(window=wsize, min_periods=1, center=True).mean()
    mask = []
    for r in y_mean:
        if r <= threshold:
            mask.append(False)
        else:
            mask.append(True)
    return np.array(mask)



""" features """
def get_melspectrogram(signal, config):
    spec = librosa.feature.melspectrogram(
        y=signal,
        sr=config.sr,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_mels = config.n_mels,
        fmin = config.fmin,
        fmax = config.fmax
    )
    spec = librosa.amplitude_to_db(spec)

    return spec
