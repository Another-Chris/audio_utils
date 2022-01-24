from polyAu.preprocessing import AudioFeatureGenerator,create_fdirs
import matplotlib.pyplot as plt
import pandas as pd
from handle_folder_structure import handle_meta

valid_dir = "./kaggle_audio_tagging/valid"
dirs = create_fdirs(valid_dir)
fnames = dirs[0]
labels = dirs[1]

train_meta = handle_meta("./kaggle_audio_tagging/meta/train.csv")


for i in range(len(fnames)):
    fname = fnames[i]
    label = labels[i]

    exp_label = str(next((row[0] for row in train_meta if row[1] in fname),None))
    if label != exp_label:
        print("wrong!")





# afg = AudioFeatureGenerator(
#     feature_type = "mel_spectrogram",
#     resolution_range = [1,2,4],
# )
# train_gen = afg.flow_from_directory("./kaggle_audio_tagging/train")
# data, label = next(train_gen)
