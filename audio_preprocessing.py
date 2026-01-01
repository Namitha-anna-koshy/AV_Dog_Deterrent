#Mount your drive
from google.colab import drive
drive.mount('/content/drive')

#aggressive audio to spectrogram

import os
import librosa
import numpy as np

# ===== PATHS =====
input_dir = "/content/drive/MyDrive/Audio_dataset/aggressive"
output_dir = "/content/drive/MyDrive/Audio_training/aggressive"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

sample_rate = 16000
target_duration = 5.0   # seconds fixed length
target_samples = int(sample_rate * target_duration)

# ===== FUNCTION: Audio → Log-Mel =====
def audio_to_logmel(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate)

    # pad or trim to same shape
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

# ===== CONVERT LOOP =====
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(input_dir, filename)
        logmel = audio_to_logmel(file_path)

        save_path = os.path.join(output_dir, filename.replace(".wav", ".npy"))
        np.save(save_path, logmel)
        print(f"Saved: {save_path}")

print("\n🎉 Conversion Completed Successfully!")

#Converting non agressive
from google.colab import drive
drive.mount('/content/drive')

import os
import librosa
import numpy as np

# ===== PATHS =====
input_dir = "/content/drive/MyDrive/Audio_dataset/non aggressive"
output_dir = "/content/drive/MyDrive/Audio_training/non_aggressive"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

sample_rate = 16000
target_duration = 5.0   # seconds fixed length
target_samples = int(sample_rate * target_duration)

# ===== FUNCTION: Audio → Log-Mel =====
def audio_to_logmel(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate)

    # pad or trim to same shape
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

# ===== CONVERT LOOP =====
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(input_dir, filename)
        logmel = audio_to_logmel(file_path)

        save_path = os.path.join(output_dir, filename.replace(".wav", ".npy"))
        np.save(save_path, logmel)
        print(f"Saved: {save_path}")

print("\n🎉 Conversion Completed Successfully!")
