import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *

os.makedirs(mel_path, exist_ok=True)
metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"), sep="|", header=None)

for fname, text in tqdm(metadata[[0,1]].values):
    wav_path = os.path.join(data_path, "wavs", f"{fname}.wav")
    y, sr = librosa.load(wav_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, win_length=win_length, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.clip((mel + 100) / 100, 0, 1)
    np.save(os.path.join(mel_path, f"{fname}.npy"), mel)

print("✅ Done preprocessing audio to mel-spectrogram")