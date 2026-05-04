#force to merge 

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pickle
from ultralytics import YOLO
import os

class DogDefenseSystem:
    def __init__(self):
        print("⏳ Initializing Defense Systems...")
        
        self.visual_model = tf.keras.models.load_model('dog_aggression_model.h5')
        self.yolo = YOLO('yolov8n.pt')

        self.yamnet = hub.load("yamnet")
        self.audio_cnn = None
        self.audio_pkl = None
        self.feat_ext = None

        if os.path.exists('audio_cnn_model.h5'):
            self.audio_cnn = tf.keras.models.load_model('audio_cnn_model.h5')
            try:
                self.feat_ext = tf.keras.Model(inputs=self.audio_cnn.input, outputs=self.audio_cnn.get_layer("feature_output").output)
            except:
                self.feat_ext = tf.keras.Model(inputs=self.audio_cnn.input, outputs=self.audio_cnn.layers[-2].output)

        if os.path.exists('audio_ensemble_classifier.pkl'):
            with open('audio_ensemble_classifier.pkl', "rb") as f:
                self.audio_pkl = pickle.load(f)

        self.ALPHA = 0.60
        self.BETA = 0.40
        self.THRESHOLD = 0.45 
        
        # --- The Memory Latch ---
        self.latch_counter = 0 
        
        print("✅ Models Loaded & System Ready!")

    def reset(self):
        """Wipes the AI's memory between video switches"""
        self.latch_counter = 0
        self.latest_data = {"score": 0.0, "status": "SAFE"}

    def analyze_audio(self, audio_chunk, sr=16000):
        if len(audio_chunk) == 0: return 0.0
        scores, embeddings, _ = self.yamnet(audio_chunk)
        dog_conf = np.mean(scores.numpy()[:, 70:85]) 
        
        if dog_conf < 0.05: return 0.0 

        if self.audio_cnn:
            target_len = int(sr * 5.0)
            if len(audio_chunk) < target_len:
                audio_chunk = np.pad(audio_chunk, (0, target_len - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:target_len]
            
            mel = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=64)
            logmel = librosa.power_to_db(mel, ref=np.max)
            inp = np.expand_dims(logmel, axis=(0, -1))
            
            cnn_prob = float(self.audio_cnn.predict(inp, verbose=0)[0][0])
            pkl_prob = cnn_prob 
            
            if self.audio_pkl and self.feat_ext:
                try:
                    features = self.feat_ext.predict(inp, verbose=0).reshape(1, -1)
                    pkl_prob = float(self.audio_pkl.predict_proba(features)[0][1])
                except:
                    pass
            
            return (cnn_prob + pkl_prob) / 2
        return 0.0

    def predict(self, frame, audio_chunk=None):
        #  classes= [16] for real dogs
        # conf=0.15 forces YOLO to detect the dog even on a low-quality phone screen
        results = self.yolo(frame, classes=[16], verbose=False)
        dog_detected = False
        p_visual = 0.0
        display_frame = frame.copy()

        # --- 1. INITIALIZE MEMORY (Runs only on the first frame) ---
        if not hasattr(self, 'last_box'):
            self.last_box = (0, 0, 0, 0)
            self.box_timer = 0

        # --- 2. UPDATE MEMORY IF TARGET IS SEEN ---
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            self.last_box = tuple(map(int, box.xyxy[0])) # Lock the coordinates
            self.box_timer = 5 # Remember this location for the next 5 frames

        # --- 3. USE MEMORY TO DRAW AND PREDICT ---
        if self.box_timer > 0:
            dog_detected = True
            x1, y1, x2, y2 = self.last_box
            self.box_timer -= 1 # Countdown the timer
            
            # Draw the stabilized box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Keep predicting visual threat on this locked area
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                img = cv2.resize(crop, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.astype('float32') / 255.0
                p_visual = float(self.visual_model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0])

        p_audio = 0.0
        if audio_chunk is not None:
            p_audio = self.analyze_audio(audio_chunk)

        if p_audio < 0.05:
            final_threat = p_visual * 0.9 
        else:
            final_threat = (self.ALPHA * p_visual) + (self.BETA * p_audio)
        print(f"Live X-Ray -> Vision: {p_visual:.2f} | Audio: {p_audio:.2f} | Final: {final_threat:.2f} | Dog Seen: {dog_detected}")    
        is_aggressive = (final_threat > self.THRESHOLD) and dog_detected
        
        # --- LATCH LOGIC APPLIED HERE ---
        if is_aggressive:
            self.latch_counter = 15 

        display_threat = False
        if self.latch_counter > 0:
            display_threat = True
            self.latch_counter -= 1
            final_threat = max(final_threat, self.THRESHOLD + 0.1)

        # --- ALERTS ---
        status = "SAFE"
        if display_threat:
            status = "AGGRESSIVE"
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 20)
            cv2.putText(display_frame, "THREAT DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return display_frame, {
            "score": round(final_threat * 100, 1), 
            "status": status
        }