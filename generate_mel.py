import torch
import numpy as np
import os
from model.tacotron2 import Tacotron2
from utils import text_to_sequence
from config import *


def generate_mel(text, ckpt_path="checkpoints/tacotron2_best.pth", out_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Tacotron2(vocab_size=256).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    seq = torch.LongTensor(text_to_sequence(text)).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([seq.shape[1]]).to(device)
    mel_inp = torch.zeros(1, 1, n_mels).to(device)

    with torch.no_grad():
        _, mel_post, _, _ = model(seq, mel_inp, text_lengths)

    mel_np = mel_post.squeeze(0).cpu().numpy()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mel_np)
        print(f"✅ Mel saved to {out_path}")
    else:
        print("ℹ️ No output path provided, returning mel array.")

    return mel_np
