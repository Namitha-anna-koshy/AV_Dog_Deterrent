import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import threading
import time
import librosa
import joblib
import pickle
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# ⚠️ UPDATE THESE FILENAMES TO MATCH YOURS
VISUAL_MODEL_PATH = 'dog_aggression_model.h5' 
AUDIO_CNN_PATH = 'audio_cnn_model.h5'
AUDIO_ENSEMBLE_PATH = 'audio_ensemble_classifier.pkl' # Optional

# Audio Settings
SAMPLE_RATE = 16000 # YAMNet requires 16k
DURATION = 2.0      # Listen in 2-second chunks
ULTRASONIC_FREQ = 15000

# Fusion Weights (From your Methodology)
ALPHA = 0.8 # Visual Weight
BETA = 0.2  # Audio Weight
THRESHOLD = 0.6

# ==========================================
# 2. AUDIO LISTENER (The "Ears")
# ==========================================
class AudioSystem:
    def __init__(self):
        self.score = 0.0
        self.status = "Initializing..."
        self.running = True
        self.models_loaded = False
        
        print("⏳ Loading Audio Models (YAMNet + Custom)...")
        try:
            # 1. Load YAMNet (from TFHub)
            self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            
            # 2. Load Your Custom CNN
            if os.path.exists(AUDIO_CNN_PATH):
                self.cnn_model = tf.keras.models.load_model(AUDIO_CNN_PATH)
                self.models_loaded = True
                print("✅ Audio Models Loaded!")
            else:
                print(f"⚠️ Warning: {AUDIO_CNN_PATH} not found. Audio will be ignored.")
                
        except Exception as e:
            print(f"❌ Error loading audio models: {e}")
            self.models_loaded = False

    def audio_to_logmel(self, y):
        # Must match your training logic exactly
        if len(y) < int(SAMPLE_RATE * 5.0): # Pad if too short
            y = np.pad(y, (0, int(SAMPLE_RATE * 5.0) - len(y)))
        else:
            y = y[:int(SAMPLE_RATE * 5.0)] # Crop if too long
            
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
        return librosa.power_to_db(mel, ref=np.max)

    def start_listening(self):
        if not self.models_loaded: return
        # Run loop in a separate thread
        t = threading.Thread(target=self._loop)
        t.daemon = True
        t.start()

    def _loop(self):
        print("🎙️ Microphone Active (Listening in background...)")
        while self.running:
            try:
                # 1. Record 2 Seconds
                audio_data = sd.rec(int(DURATION * SAMPLE_RATE), 
                                  samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                sd.wait() # Wait for recording to finish
                audio_data = audio_data.flatten()

                # 2. Check YAMNet (Is it a dog?)
                scores, embeddings, _ = self.yamnet(audio_data)
                # Class 70-80 roughly covers dog sounds in YAMNet
                dog_score = np.mean(scores.numpy()[:, 70:85]) 
                
                if dog_score < 0.05:
                    self.score = 0.0
                    self.status = "Silence/Noise"
                else:
                    # 3. Check Aggression (CNN)
                    logmel = self.audio_to_logmel(audio_data)
                    inp = np.expand_dims(logmel, axis=(0, -1))
                    
                    # Predict
                    pred = self.cnn_model.predict(inp, verbose=0)[0][0]
                    self.score = float(pred)
                    self.status = f"Dog Detected ({pred:.1%})"
                    
            except Exception as e:
                print(f"Audio Error: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False

# ==========================================
# 3. ULTRASONIC ACTUATOR (The "Mouth")
# ==========================================
class UltrasonicSystem:
    def __init__(self):
        self.active = False
        self.stop_event = threading.Event()
        self.thread = None

    def _play_sound(self):
        # Continuous generation
        with sd.OutputStream(channels=1, samplerate=44100) as stream:
            t = 0
            while not self.stop_event.is_set():
                frames = 4410
                time_arr = (np.arange(frames) + t) / 44100
                wave = 0.5 * np.sin(2 * np.pi * ULTRASONIC_FREQ * time_arr)
                stream.write(wave.astype(np.float32))
                t += frames

    def start(self):
        if not self.active:
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._play_sound)
            self.thread.start()
            self.active = True

    def stop(self):
        if self.active:
            self.stop_event.set()
            self.active = False

# ==========================================
# 4. MAIN SYSTEM
# ==========================================
def main():
    # --- Load Visual Model ---
    print(f"⏳ Loading Visual Model: {VISUAL_MODEL_PATH}...")
    try:
        visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)
        print("✅ Visual Model Loaded!")
    except:
        print("❌ Visual Model Not Found.")
        return

    # --- Start Subsystems ---
    audio_sys = AudioSystem()
    audio_sys.start_listening()
    
    sound_sys = UltrasonicSystem()
    cap = cv2.VideoCapture(0)

    print("\n🛡️ MULTIMODAL SYSTEM ONLINE")
    print("   Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Visual Prediction
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype('float32') / 255.0
        img_batch = np.expand_dims(img, axis=0)

        vis_pred = visual_model.predict(img_batch, verbose=0)[0][0]
        # Assuming 0=Aggressive, 1=Safe. Invert so 1.0 = Aggression probability
        p_visual = 1.0 - vis_pred 

        # 2. Get Latest Audio Score (From background thread)
        p_audio = audio_sys.score
        
        # 3. Fusion Logic (Equation 1)
        # If visual is VERY confident, it overrides audio
        final_threat = (ALPHA * p_visual) + (BETA * p_audio)

        # 4. Decision
        is_aggressive = final_threat > THRESHOLD
        
        # 5. UI & Actuation
        color = (0, 0, 255) if is_aggressive else (0, 255, 0)
        status_text = "AGGRESSIVE" if is_aggressive else "SAFE"
        
        if is_aggressive:
            sound_sys.start()
            cv2.putText(frame, "🔊 ULTRASONIC EMITTING", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        else:
            sound_sys.stop()

        # Dashboard
        cv2.rectangle(frame, (0,0), (640, 80), (50,50,50), -1)
        cv2.putText(frame, f"VISUAL: {p_visual:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(frame, f"AUDIO:  {p_audio:.2f} ({audio_sys.status})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(frame, f"FUSION: {final_threat:.2f}", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow('Multimodal Dog Defense', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    audio_sys.stop()
    sound_sys.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()