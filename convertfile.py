import os
import shutil
import pandas as pd

SRC_WAV_DIR = "data/LJSpeech-1.1/wavs"
SRC_MEL_DIR = "data/mels"
DST_WAV_DIR = "trainset_hifigan/wavs"
DST_MEL_DIR = "trainset_hifigan/mels"

os.makedirs(DST_WAV_DIR, exist_ok=True)
os.makedirs(DST_MEL_DIR, exist_ok=True)

metadata = pd.read_csv("data/LJSpeech-1.1/metadata.csv", sep="|", header=None)
for i, (wav_id, _) in enumerate(metadata[[0, 1]].values):
    if i >= 7000: break
    wav_file = f"{wav_id}.wav"
    mel_file = f"{wav_id}.npy"

    src_wav_path = os.path.join(SRC_WAV_DIR, wav_file)
    src_mel_path = os.path.join(SRC_MEL_DIR, mel_file)
    dst_wav_path = os.path.join(DST_WAV_DIR, wav_file)
    dst_mel_path = os.path.join(DST_MEL_DIR, mel_file)

    if os.path.exists(src_wav_path) and os.path.exists(src_mel_path):
        shutil.copyfile(src_wav_path, dst_wav_path)
        shutil.copyfile(src_mel_path, dst_mel_path)

print("✅ Copied 7000 wav + mel files to trainset_hifigan/")
