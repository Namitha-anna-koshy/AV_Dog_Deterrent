import cv2
import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import pickle
import librosa
import threading
import time
import os
from scipy.io.wavfile import write

# ==============================================================================
# 1. SETUP & PATHS
# ==============================================================================
print("🔄 Initializing Systems... (This may take a moment)")

# AUDIO SETTINGS
SAMPLE_RATE = 16000
DURATION = 3 # Seconds to record when triggered
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

# ⚠️ PATH TO YOUR FOLDER - Ensured this matches your local structure
MODEL_DIR = r"C:\Users\DELL\OneDrive\Desktop\Projects\AV_Dog_Deterrent"

IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "dog_aggression_model.h5")
CNN_PATH = os.path.join(MODEL_DIR, "audio_cnn_model.h5")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "audio_ensemble_classifier.pkl")

# ==============================================================================
# 2. LOAD MODELS
# ==============================================================================
try:
    # A. Visual Model
    print(f"⏳ Loading Visual Model from: {IMAGE_MODEL_PATH}")
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise FileNotFoundError(f"Missing file: {IMAGE_MODEL_PATH}")
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    
    # B. Audio Models
    print(f"⏳ Loading Audio Models from: {CNN_PATH}")
    if not os.path.exists(CNN_PATH) or not os.path.exists(ENSEMBLE_PATH):
        raise FileNotFoundError("Missing Audio Model files (.h5 or .pkl)")
        
    cnn_model = tf.keras.models.load_model(CNN_PATH)
    ensemble = pickle.load(open(ENSEMBLE_PATH, "rb"))
    
    # Feature Extractor for Ensemble
    feature_model = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("feature_output").output
    )

    # C. YAMNet (Dog Detector)
    print("⏳ Loading YAMNet...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # Load Class Names
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    class_names = np.loadtxt(class_map_path, dtype=str, delimiter=',', skiprows=1, usecols=2)
    
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    print("Please check that the file names in your folder match exactly!")
    exit()

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def audio_to_logmel(y, sr):
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    return librosa.power_to_db(mel, ref=np.max)

def get_visual_score(frame):
    # Resize to 224x224
    img_resized = tf.image.resize(frame, [224, 224])
    # Convert to Grayscale -> Back to RGB (Matches your training)
    img_gray = tf.image.rgb_to_grayscale(img_resized)
    img_rgb = tf.image.grayscale_to_rgb(img_gray)
    img_tensor = np.expand_dims(img_rgb, axis=0)
    
    # Predict
    preds = image_model.predict(img_tensor, verbose=0)
    # Assuming 0=Aggressive, 1=Non-Aggressive
    return 1.0 - preds[0][0]

def analyze_snapshot(frame, audio_data):
    """
    Real-World Smart Logic for B.Tech Major Project.
    """
    print("\n🔍 Analyzing Snapshot...")
    
    # --- 1. VISUAL ANALYSIS ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    visual_prob = get_visual_score(frame_rgb)
    
    # --- 2. AUDIO ANALYSIS ---
    y = audio_data.flatten()
    
    # A. YAMNet Check
    scores, embeddings, spectrogram = yamnet_model(y)
    dog_keywords = ["Dog", "Bark", "Bow-wow", "Woof", "Growling", "Yelp", "Howl"]
    dog_indices = [i for i, name in enumerate(class_names) if any(k.lower() in name.lower() for k in dog_keywords)]
    
    dog_scores = tf.gather(scores, indices=dog_indices, axis=1)
    dog_mean = float(np.mean(dog_scores))
    
    audio_prob = 0.0
    audio_status = "Silent"
    is_dog_audio_present = False

    if dog_mean > 0.10: 
        is_dog_audio_present = True
        logmel = audio_to_logmel(y, SAMPLE_RATE)
        inp = np.expand_dims(logmel, axis=(0, -1))
        
        cnn_prob = float(cnn_model.predict(inp, verbose=0)[0][0])
        embedding = feature_model.predict(inp, verbose=0).reshape(1, -1)
        ensemble_prob = float(ensemble.predict_proba(embedding)[0][1])
        
        audio_prob = (cnn_prob + ensemble_prob) / 2
        audio_status = f"Dog Detected ({audio_prob:.1%})"
    else:
        audio_status = "Ignored (Silence)"

    # --- 3. SMART REAL-WORLD LOGIC ---
    if is_dog_audio_present and audio_prob > 0.60:
        final_score = audio_prob
        logic_msg = "🔊 Audio Priority (Hidden Dog Logic)"
    elif not is_dog_audio_present:
        if visual_prob > 0.80:
            final_score = visual_prob
            logic_msg = "👁️ High Visual Trust (Silent Threat)"
        else:
            final_score = visual_prob * 0.5 
            logic_msg = "mute Silence Penalty (Ambiguous Visual)"
    else:
        final_score = max(visual_prob, audio_prob)
        logic_msg = "⚠️ Multi-Modal Confirmation"

    label = "AGGRESSIVE" if final_score > 0.50 else "NON-AGGRESSIVE"
    
    print("-" * 40)
    print(f"👁️ Visual Score: {visual_prob:.1%}")
    print(f"🔊 Audio Score:  {audio_prob:.1%} ({audio_status})")
    print(f"🧠 Logic:         {logic_msg}")
    print(f"⚡ FINAL SCORE:  {final_score:.1%}")
    print(f"🏆 PREDICTION:   {label}")
    print("-" * 40)
    
    # FIXED: Returning 4 values to match the unpack in main()
    return label, final_score, visual_prob, audio_prob

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
def main():
    print("\n🔍 SEARCHING FOR WEBCAMS...")
    cap = None
    
    for i in range(2):
        print(f"   👉 Testing Camera Index {i}...")
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            print(f"   ✅ SUCCESS! Found camera at Index {i}")
            cap = temp_cap
            break
        else:
            temp_cap.release()

    if cap is None or not cap.isOpened():
        print("\n❌ CRITICAL ERROR: No camera found.")
        return

    print("\n🎥 CAMERA READY!")
    print("👉 Press 'SPACEBAR' to Record (3s) & Analyze.")
    print("👉 Press 'Q' to Quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press SPACE to Analyze", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Dog Aggression Detector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # Spacebar
            print(f"🎤 Debug: Recording from {sd.query_devices(kind='input')['name']}")
            print("\n🔴 RECORDING 3 SECONDS...")
            myrecording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            
            start_time = time.time()
            captured_frame = frame.copy()
            
            while (time.time() - start_time) < DURATION:
                ret, frame = cap.read()
                cv2.putText(frame, "RECORDING AUDIO...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Dog Aggression Detector', frame)
                cv2.waitKey(1)
            
            sd.wait() 
            print("✅ Capture Complete. Processing...")
            
            # FIXED: Unpacking 4 values correctly
            label, score, v_prob, a_prob = analyze_snapshot(captured_frame, myrecording)
            
            # --- DISPLAY LOGIC (Indented inside Spacebar block) ---
            color = (0, 0, 255) if label == "AGGRESSIVE" else (0, 255, 0)
            
            # Check for silent threat safety gate
            display_label = label
            if v_prob > 0.85 and a_prob == 0:
                print("💡 Shield Active: Verifying silent high-visual threat...")
            
            cv2.rectangle(captured_frame, (0, 0), (640, 80), (0,0,0), -1)
            cv2.putText(captured_frame, f"{display_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(captured_frame, f"Conf: {score:.1%}", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.imshow('Dog Aggression Detector', captured_frame)
            cv2.waitKey(3000)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()