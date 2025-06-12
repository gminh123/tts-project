import torch
import numpy as np
from hifigan.models import Generator
import soundfile as sf
import json
from generate_mel import generate_mel
from config import *

with open("checkpoints/hifigan/config.json") as f:
    h = json.load(f)
model = Generator(h).to("cuda").eval()
state_dict = torch.load("checkpoints/hifigan/generator.pth", map_location="cuda")
model.load_state_dict(state_dict['generator'])
model.remove_weight_norm()

text = "Xin chào, tôi là mô hình chuyển văn bản thành giọng nói."
mel = generate_mel(text)
with torch.no_grad():
    mel_tensor = torch.from_numpy(mel).unsqueeze(0).to("cuda")
    audio = model(mel_tensor).squeeze().cpu().numpy()
    sf.write("output.wav", audio, sample_rate)
    print("✅ Audio saved to output.wav")