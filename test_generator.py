import tensorflow as tf
from preprocessing import AudioFeatureGenerator,create_fdirs
import os
import matplotlib.pyplot as plt
import cv2

# fdirs = create_fdirs("./kaggle_audio_tagging/train")



root_dir = "./kaggle_audio_tagging"
train_dir = root_dir + "/train"
valid_dir = root_dir + "/valid"
train_size = len(create_fdirs(train_dir)[0])
valid_size = len(create_fdirs(valid_dir)[0])
batch_size = 32

train_gen = AudioFeatureGenerator(
    feature_type = "mel_spectrogram",
    num_mels = 128,
    resolution_range = [1,2,4],
)
train_generator = train_gen.flow_from_directory(
     train_dir,
     fix_length = True,
     batch_size = batch_size)


valid_gen =  AudioFeatureGenerator(
    feature_type = "mel_spectrogram",
    num_mels = 128,
    resolution_range = [1,2,4],
)
valid_generator = valid_gen.flow_from_directory(valid_dir, batch_size = batch_size)



for i,dl in enumerate(train_generator):
    image = dl[0]
    label = dl[1]
    print(image.shape)
    print(label)
    break
