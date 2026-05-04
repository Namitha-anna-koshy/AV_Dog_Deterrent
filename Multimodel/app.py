import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import os
import tempfile
import winsound
import pickle
import time
from moviepy import VideoFileClip
from ultralytics import YOLO

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Canine Aggression Defense", page_icon="🐕", layout="wide")
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
    h1 { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    # File Paths
    VISUAL_PATH = 'dog_aggression_model.h5' 
    AUDIO_CNN_PATH = 'audio_cnn_model.h5'
    AUDIO_PKL_PATH = 'audio_ensemble_classifier.pkl'
    YOLO_NAME = 'yolov8n.pt'
    
    status = st.empty()
    status.info("⏳ Initializing Defense Systems...")

    # Load Visual
    try:
        visual_model = tf.keras.models.load_model(VISUAL_PATH)
        yolo_model = YOLO(YOLO_NAME)
    except Exception as e:
        st.error(f"❌ Critical Error: {e}")
        return None, None, None, None, None, None

    # Load Audio
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    cnn_model = None
    feature_extractor = None
    ensemble_model = None

    if os.path.exists(AUDIO_CNN_PATH):
        try:
            cnn_model = tf.keras.models.load_model(AUDIO_CNN_PATH)
            # Create feature extractor for pickle model
            try:
                feature_extractor = tf.keras.Model(
                    inputs=cnn_model.input, 
                    outputs=cnn_model.get_layer("feature_output").output
                )
            except:
                feature_extractor = tf.keras.Model(
                    inputs=cnn_model.input, 
                    outputs=cnn_model.layers[-2].output
                )
        except:
            pass

    if os.path.exists(AUDIO_PKL_PATH):
        try:
            with open(AUDIO_PKL_PATH, "rb") as f:
                ensemble_model = pickle.load(f)
        except:
            pass

    status.empty()
    return visual_model, yolo_model, yamnet, cnn_model, feature_extractor, ensemble_model

visual_model, yolo_model, yamnet_model, audio_cnn, feat_ext, audio_pkl = load_models()

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================
def analyze_audio_smart(audio_data, yamnet, cnn, extractor, pkl_model, sr=16000):
    scores, embeddings, _ = yamnet(audio_data)
    dog_conf = np.mean(scores.numpy()[:, 70:85]) 
    
    if dog_conf < 0.05: return 0.0, "Silence"

    if cnn:
        target_len = int(sr * 5.0)
        if len(audio_data) < target_len:
            audio_data = np.pad(audio_data, (0, target_len - len(audio_data)))
        else:
            audio_data = audio_data[:target_len]
        
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=64)
        logmel = librosa.power_to_db(mel, ref=np.max)
        inp = np.expand_dims(logmel, axis=(0, -1))
        
        cnn_prob = float(cnn.predict(inp, verbose=0)[0][0])
        
        pkl_prob = cnn_prob
        if pkl_model and extractor:
            try:
                features = extractor.predict(inp, verbose=0).reshape(1, -1)
                pkl_prob = float(pkl_model.predict_proba(features)[0][1])
            except:
                pass
        
        final_score = (cnn_prob + pkl_prob) / 2
        return final_score, f"Bark Detected ({final_score:.1%})"
    
    return 0.0, "Detected"

# ==========================================
# 4. TUNED SETTINGS (CLIENT READY)
# ==========================================
st.sidebar.title("🛡️ System Control")
st.sidebar.success("System Status: ONLINE")

# --- LOCKED SETTINGS (No messing around) ---
# We trust Audio more now (40%) to catch the barking dog
# We lower Visual trust (60%) so the happy dog stays safe
ALPHA = 0.60 
BETA = 0.40
THRESHOLD = 0.45 

show_yolo = st.sidebar.checkbox("Show Detection Box", value=True)

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("🛡️ AV Dog Defense System")
input_mode = st.radio("Select Source:", ["Upload Video", "Live Camera"], horizontal=True)

start_button = False
cap = None
temp_audio_path = None
audio_full = None
is_file = False

if input_mode == "Upload Video":
    uploaded_file = st.file_uploader("Select MP4/AVI", type=['mp4', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        is_file = True
        
        with st.spinner("Preparing Analysis..."):
            try:
                vc = VideoFileClip(video_path)
                temp_audio_path = "temp_audio_st.wav"
                vc.audio.write_audiofile(temp_audio_path, fps=16000, logger=None)
                audio_full, sr = librosa.load(temp_audio_path, sr=16000)
            except:
                audio_full = np.zeros(16000)
                sr = 16000
        
        cap = cv2.VideoCapture(video_path)
        start_button = st.button("▶️ RUN ANALYSIS")

elif input_mode == "Live Camera":
    if st.button("🎥 ACTIVATE CAMERA"):
        cap = cv2.VideoCapture(0)
        start_button = True

# ==========================================
# 6. EXECUTION LOOP
# ==========================================
if start_button and cap and visual_model:
    st_frame = st.empty()
    kpi1, kpi2, kpi3 = st.columns(3)
    warning_placeholder = st.empty()
    
    current_frame = 0
    fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 30
    
    # Latch Variables (To prevent flickering)
    threat_latch_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_frame += 1
        
        # --- 1. YOLO ---
        results = yolo_model(frame, classes=[16], verbose=False)
        dog_detected = False
        p_visual = 0.0
        
        display_frame = frame.copy()
        
        if len(results[0].boxes) > 0:
            dog_detected = True
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if show_yolo:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                img = cv2.resize(crop, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.astype('float32') / 255.0
                
                raw_pred = visual_model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
                
                # --- FINAL LOGIC: STANDARD ---
                # We use raw_pred because "Unchecked" (Standard) worked for the Happy Dog
                p_visual = raw_pred 
                # -----------------------------

        # --- 2. Audio ---
        p_audio = 0.0
        if is_file and audio_full is not None:
            t_sec = current_frame / fps
            start = int(t_sec * sr)
            end = start + int(2.0 * sr)
            if end < len(audio_full):
                chunk = audio_full[start:end]
                p_audio, _ = analyze_audio_smart(chunk, yamnet_model, audio_cnn, feat_ext, audio_pkl)
        
        # --- 3. Weighted Fusion ---
        if p_audio < 0.05:
            # If silent, rely on visual but scale it down slightly to be safe
            final_threat = p_visual * 0.9 
        else:
            # If barking, let Audio have 40% influence
            final_threat = (ALPHA * p_visual) + (BETA * p_audio)
            
        is_aggressive = (final_threat > THRESHOLD) and dog_detected
        
        # --- 4. Latch Logic (Smooths the output) ---
        if is_aggressive:
            threat_latch_counter = 10 # Stay red for ~10 frames (0.3s) even if signal dips
        
        display_threat = False
        if threat_latch_counter > 0:
            display_threat = True
            threat_latch_counter -= 1

        # --- 5. UI Updates ---
        kpi1.metric("Visual Threat", f"{p_visual:.1%}")
        kpi2.metric("Audio Threat", f"{p_audio:.1%}")
        kpi3.metric("Fusion Score", f"{final_threat:.1%}")

        if display_threat:
            warning_placeholder.error("🚨 THREAT DETECTED! ULTRASONIC EMITTING 🔊")
            cv2.rectangle(display_frame, (0,0), (display_frame.shape[1], display_frame.shape[0]), (0,0,255), 20)
            
            try:
                # Use 12,000 Hz: It's high-pitched and "annoying" but guaranteed to be heard
                winsound.Beep(12000, 500) 
            except:
                pass