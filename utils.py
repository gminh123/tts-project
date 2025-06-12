import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from config import mel_path
import os

def text_to_sequence(text):
    return [ord(c) if ord(c) < 256 else 0 for c in text]

class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, limit=None):
        self.items = []
        with open(metadata_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                name, text = line.strip().split("|")[:2]
                self.items.append((name, text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, text = self.items[idx]
        mel = np.load(f"{mel_path}/{fname}.npy")
        if mel.shape[1] > 1000:
            return None
        return torch.LongTensor(text_to_sequence(text)), torch.FloatTensor(mel.T)

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    texts, mels = zip(*batch)
    text_lens = torch.LongTensor([len(t) for t in texts])
    mel_lens = torch.LongTensor([m.shape[0] for m in mels])
    return (
        pad_sequence(texts, batch_first=True),
        text_lens,
        pad_sequence(mels, batch_first=True),
        mel_lens
    )