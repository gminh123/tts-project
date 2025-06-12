from generate_mel import generate_mel
import matplotlib.pyplot as plt

# Test văn bản
text = "I am a Tacotron2 model."

mel = generate_mel(
    text=text,
    ckpt_path="checkpoints/tacotron2_best.pth"
)

# Hiển thị mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel.T, aspect="auto", origin="lower", interpolation="none")
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel bins")
plt.colorbar()
plt.tight_layout()
plt.show()
