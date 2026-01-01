#predictions using noisy data
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os

# ====== Load trained models ======
cnn_model = tf.keras.models.load_model("/content/drive/MyDrive/aggression_cnn_model.h5")
ensemble = pickle.load(open("/content/drive/MyDrive/ensemble_classifier.pkl", "rb"))

# ====== Prediction parameters ======
sample_rate = 16000
target_duration = 5.0
target_samples = int(sample_rate * target_duration)

def audio_to_logmel(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate)

    # pad or trim
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel

def predict_audio(file_path, threshold=0.5):
    logmel = audio_to_logmel(file_path)

    # reshape for CNN
    inp = np.expand_dims(logmel, axis=(0, -1))   # (1,64,157,1)

    # CNN prediction probability
    cnn_prob = cnn_model.predict(inp)[0][0]

    # Extract embeddings from CNN
    feature_model = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("feature_output").output
    )
    embedding = feature_model.predict(inp).reshape(1,-1)

    # Ensemble prediction
    ensemble_prob = ensemble.predict_proba(embedding)[0][1]

    # Final weighted avg (optional)
    final_prob = (cnn_prob + ensemble_prob) / 2

    print("CNN Probability:", cnn_prob)
    print("Ensemble Probability:", ensemble_prob)
    print("Final Probability:", final_prob)

    if final_prob > threshold:
        return "⚠️ Aggressive"
    else:
        return "🙂 Non-aggressive"


# ======== Example Use ========
file = "/content/drive/MyDrive/Audio_training/test_audio/d.wav"
result = predict_audio(file)
print("\nPrediction:", result)
