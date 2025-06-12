import numpy as np
import matplotlib.pyplot as plt
import sys

mel_path = sys.argv[1]  # truyền đường dẫn file .npy qua dòng lệnh
mel = np.load(mel_path)

plt.figure(figsize=(10, 4))
plt.imshow(mel, aspect='auto', origin='lower')
plt.title("Mel Spectrogram")
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Mel bins")
plt.tight_layout()
plt.show()