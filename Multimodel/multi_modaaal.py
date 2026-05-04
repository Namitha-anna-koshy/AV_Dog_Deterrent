import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import threading
import time
import librosa
import os
import tkinter as tk
from tkinter import filedialog
from moviepy import VideoFileClip
from ultralytics import YOLO  # <--- NEW: Import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
# ⚠️ CHECK FILENAMES
VISUAL_MODEL_PATH = 'dog_aggression_model.h5' 
AUDIO_CNN_PATH = 'audio_cnn_model.h5'
YOLO_MODEL_NAME = 'yolov8n.pt' # 'n' = Nano (Fastest for laptops)

# Fusion Weights
ALPHA = 0.8 # Visual Weight
BETA = 0.2  # Audio Weight
THRESHOLD = 0.6

# ==========================================
# 2. AUDIO ANALYZER
# ==========================================
class AudioAnalyzer:
    def __init__(self):
        print("⏳ Loading Audio Models...")
        # Load YAMNet from TFHub
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        
        # Load Custom CNN
        if os.path.exists(AUDIO_CNN_PATH):
            self.cnn_model = tf.keras.models.load_model(AUDIO_CNN_PATH)
            print("✅ Audio CNN Loaded!")
        else:
            self.cnn_model = None
            print("⚠️ Audio CNN not found. Using YAMNet only.")

    def analyze_audio_segment(self, audio_data, sample_rate=16000):
        # A. YAMNet Check (Is it a dog?)
        scores, embeddings, _ = self.yamnet(audio_data)
        # Check indices 70-85 (Dog, Bark, Growl, Howl)
        dog_confidence = np.mean(scores.numpy()[:, 70:85]) 
        
        if dog_confidence < 0.05:
            return 0.0, "Silence"
            
        # B. CNN Check (Aggression)
        if self.cnn_model:
            # 1. Pad/Crop to 5 seconds
            target_len = int(sample_rate * 5.0)
            if len(audio_data) < target_len:
                audio_data = np.pad(audio_data, (0, target_len - len(audio_data)))
            else:
                audio_data = audio_data[:target_len]
                
            # 2. Convert to Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=64)
            logmel = librosa.power_to_db(mel, ref=np.max)
            inp = np.expand_dims(logmel, axis=(0, -1))
            
            # 3. Predict
            pred = self.cnn_model.predict(inp, verbose=0)[0][0]
            return float(pred), f"Dog Audio ({pred:.1%})"
        
        return 0.0, "Dog Detected (No CNN)"

# ==========================================
# 3. FILE SELECTION
# ==========================================
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video to Test",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    return file_path

# ==========================================
# 4. MAIN SYSTEM
# ==========================================
def main():
    # --- A. Load Models ---
    print(f"⏳ Loading Visual Classifier: {VISUAL_MODEL_PATH}...")
    try:
        visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)
        print("✅ Visual Classifier Loaded!")
    except:
        print("❌ Visual Classifier Not Found.")
        return

    print(f"⏳ Loading YOLOv8 ({YOLO_MODEL_NAME})...")
    # This will download yolov8n.pt automatically on first run
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print("✅ YOLO Loaded!")

    audio_sys = AudioAnalyzer()
    
    # --- B. Select Video ---
    print("\n📂 Waiting for you to choose a video...")
    video_path = select_file()
    if not video_path: return

    print(f"🎬 Processing: {os.path.basename(video_path)}")

    # --- C. Extract Audio ---
    print("🔊 Extracting Audio track...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile("temp_audio.wav", fps=16000, logger=None)
        y_full, sr = librosa.load("temp_audio.wav", sr=16000)
    except:
        print("⚠️ No audio track found.")
        y_full = np.zeros(16000)

    # --- D. Main Processing Loop ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        current_frame += 1

        # ==========================================
        # STEP 1: YOLO DETECTION (The Gatekeeper)
        # ==========================================
        # Run YOLO on the frame. Class 16 is 'dog'.
        results = yolo_model(frame, classes=[16], verbose=False)
        
        dog_detected = False
        p_visual = 0.0
        bbox = None

        # If YOLO sees a dog...
        if len(results[0].boxes) > 0:
            dog_detected = True
            
            # Get the box with highest confidence
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            
            # CROP the image to just the dog
            # (Adding slight padding can help accuracy)
            h, w, _ = frame.shape
            pad = 10
            y1, y2 = max(0, y1-pad), min(h, y2+pad)
            x1, x2 = max(0, x1-pad), min(w, x2+pad)
            
            dog_crop = frame[y1:y2, x1:x2]

            if dog_crop.size > 0:
                # ==========================================
                # STEP 2: VISUAL CLASSIFICATION
                # ==========================================
                # 1. Resize Crop to 224x224
                img = cv2.resize(dog_crop, (224, 224))
                # 2. Preprocess (Gray -> FakeRGB -> Normalize)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.astype('float32') / 255.0
                img_batch = np.expand_dims(img, axis=0)

                # 3. Predict Aggression
                vis_pred = visual_model.predict(img_batch, verbose=0)[0][0]
                p_visual = 1.0 - vis_pred # Invert logic if needed

        # ==========================================
        # STEP 3: AUDIO ANALYSIS
        # ==========================================
        current_time = current_frame / fps
        start_sample = int(current_time * sr)
        end_sample = start_sample + int(2.0 * sr) # 2 sec window
        
        if end_sample < len(y_full):
            audio_chunk = y_full[start_sample:end_sample]
            p_audio, audio_status = audio_sys.analyze_audio_segment(audio_chunk)
        else:
            p_audio = 0.0
            audio_status = "End"

        # ==========================================
        # STEP 4: FUSION & LOGIC
        # ==========================================
        final_threat = (ALPHA * p_visual) + (BETA * p_audio)
        
        # Only trigger if threats > threshold AND a dog is visually confirmed
        is_aggressive = (final_threat > THRESHOLD) and dog_detected

        # ==========================================
        # STEP 5: VISUALIZATION
        # ==========================================
        display_img = cv2.resize(frame, (800, 600))
        
        # Draw YOLO Box (scaled to 800x600)
        if bbox:
            h_orig, w_orig = frame.shape[:2]
            sx, sy = 800/w_orig, 600/h_orig
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(display_img, (int(bx1*sx), int(by1*sy)), 
                          (int(bx2*sx), int(by2*sy)), (255, 0, 0), 3)
            cv2.putText(display_img, "YOLO DETECTED", (int(bx1*sx), int(by1*sy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Dashboard
        cv2.rectangle(display_img, (0,0), (800, 100), (30,30,30), -1)
        
        color = (0, 0, 255) if is_aggressive else (0, 255, 0)
        status_text = "AGGRESSIVE" if is_aggressive else "SAFE"
        if not dog_detected: status_text = "NO DOG"
        
        # Metrics
        cv2.putText(display_img, f"VISUAL: {p_visual:.0%}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(display_img, f"AUDIO: {p_audio:.0%} ({audio_status})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(display_img, status_text, (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

        if is_aggressive:
            cv2.putText(display_img, "🔊 ULTRASONIC EMITTING", (250, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.rectangle(display_img, (0,0), (800,600), (0,0,255), 10)

        cv2.imshow('YOLO + EfficientNet + YAMNet System', display_img)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")

if __name__ == "__main__":
    main()