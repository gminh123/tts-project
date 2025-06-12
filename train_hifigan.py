import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from model_hifigan import Generator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

class MelWavDataset(Dataset):
    def __init__(self, wav_dir, mel_dir):
        self.filenames = [f.replace(".wav", "") for f in os.listdir(wav_dir) if f.endswith(".wav")]
        self.wav_dir = wav_dir
        self.mel_dir = mel_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        wav, _ = torchaudio.load(os.path.join(self.wav_dir, name + ".wav"))
        mel = torch.from_numpy(np.load(os.path.join(self.mel_dir, name + ".npy"))).float()
        return mel, wav

with open("config_hifigan.json") as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Generator(config).to(device)
opt = optim.Adam(model.parameters(), lr=2e-4)

dataset = MelWavDataset("trainset_hifigan/wavs", "trainset_hifigan/mels")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

criterion = nn.L1Loss()

for epoch in range(100):
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for mel, wav in pbar:
        mel = mel.to(device)
        wav = wav.to(device)
        mel = mel.unsqueeze(1)
        pred = model(mel).squeeze(1)

        loss = criterion(pred, wav)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix({"loss": loss.item()})

    torch.save({"generator": model.state_dict()}, "checkpoints/hifigan/generator.pth")
    print(f"✅ Epoch {epoch} saved!")
