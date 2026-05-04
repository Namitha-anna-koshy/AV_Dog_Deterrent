# ============================================================
# Dog Aggression Deterrent System — Final Version
#
# All improvements applied:
#
#  VISUAL PIPELINE
#   - Frame resized to 480x360 before YOLO (faster)
#   - YOLO runs at imgsz=224 (faster, still accurate)
#   - Motion gate: CNN only runs when dog is moving
#   - CNN runs every 3rd frame only (reduces lag)
#   - Confidence gap: score measured above calm baseline
#   - Visual smoothing: deque(8) + 2s confirm timer
#   - Visual can trigger ALONE if strong + sustained
#
#  AUDIO PIPELINE
#   - Window reduced to 3s (faster updates ~0.9s)
#   - Slide step 0.5s (more frequent decisions)
#   - Audio smoothing: deque(6) + 1.5s confirm timer
#   - Audio can trigger ALONE if confirmed
#   - Audio absent (0.0) does NOT block visual trigger
#
#  DECISION LOGIC
#   - Visual alone triggers if confirmed + confidence > 0.30
#   - Audio alone triggers if confirmed
#   - Both weak but agreeing triggers together
#   - Status hold 2s prevents flickering
#
#  RESPONSE
#   - Deterrent tone in daemon thread (no lag)
#   - Email after 5s sustained aggression, 60s cooldown
#   - HUD: Vis, Aud, Motion, Status, Deterrent, Mail
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import sounddevice as sd
import pickle
import time
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from multiprocessing import Process, Value, Queue
from ultralytics import YOLO
from collections import deque

# ===================== EMAIL CONFIG =====================
EMAIL_SENDER           = "rithikakrishna2026@ai.sjcetpalai.ac.in"
EMAIL_PASSWORD         = "pksz pkoq xhlj chwk"       # paste Gmail App Password here
EMAIL_RECEIVER         = "rithikakrishnaau@gmail.com"
EMAIL_SUBJECT          = "⚠️ Aggressive Dog Detected"
EMAIL_TRIGGER_SECONDS  = 5
EMAIL_COOLDOWN_SECONDS = 60

# ===================== DETERRENT CONFIG =====================
DETERRENT_FREQ     = 2000     # Hz — change to 22000+ for real ultrasonic
DETERRENT_VOLUME   = 0.6
DETERRENT_SR       = 44100
DETERRENT_DURATION = 1.0

# ===================== VISUAL CONFIG =====================
CALM_BASELINE       = 0.67    # observed v_score for calm dog — measure this yourself
MOTION_THRESHOLD    = 2.0     # mean pixel diff to count as movement — tune if needed
VISUAL_BUFFER_LEN   = 8       # frames to average (covers ~1.6s at 5fps)
VISUAL_CONFIRM_TIME = 2.0     # seconds smoothed score must stay above baseline+gap
VISUAL_CONF_MIN     = 0.30    # minimum visual confidence to trigger alone
CNN_SKIP_FRAMES     = 3       # run CNN every Nth frame — reduces lag

# ===================== AUDIO CONFIG =====================
AUDIO_DEVICE_INDEX = 9        # run diagnose.py to find your mic index
AUDIO_THRESHOLD    = 0.70
AUDIO_CONFIRM_TIME = 1.5
AUDIO_BUFFER_LEN   = 6

# ===================== DECISION CONFIG =====================
# Visual triggers alone if:  visual_confirmed AND visual_confidence > VISUAL_CONF_MIN
# Audio triggers alone if:   audio_confirmed
# Both weak triggers if:     visual_confidence > 0.15 AND audio_confidence > 0.25
STATUS_HOLD_TIME    = 2.0     # seconds to stay AGGRESSIVE before reverting to NORMAL

# ===================== STATE =====================
sound_playing     = False
sound_lock        = threading.Lock()
email_sent_time   = 0.0
aggressive_since  = None
email_thread_busy = False


# ===================== DETERRENT =====================
def play_deterrent():
    """
    Generates and plays deterrent tone in a daemon thread.
    Never blocks the camera loop.
    For real deployment: set DETERRENT_FREQ=22000 and use
    an ultrasonic transducer connected to a GPIO pin.
    """
    global sound_playing
    with sound_lock:
        if sound_playing:
            return
        sound_playing = True

    def _play():
        global sound_playing
        try:
            t    = np.linspace(0, DETERRENT_DURATION,
                               int(DETERRENT_SR * DETERRENT_DURATION))
            wave = DETERRENT_VOLUME * np.sin(
                       2 * np.pi * DETERRENT_FREQ * t).astype(np.float32)
            sd.play(wave, samplerate=DETERRENT_SR)
            sd.wait()
        except Exception as e:
            print(f"[Sound] {e}")
        finally:
            sound_playing = False

    threading.Thread(target=_play, daemon=True).start()


# ===================== EMAIL =====================
def send_email_alert(duration_seconds, visual_conf, audio_score):
    """
    Sends email alert via Gmail SMTP in a daemon thread.
    To get App Password:
      Google Account → Security → 2-Step Verification → App Passwords
    """
    global email_thread_busy
    email_thread_busy = True

    def _send():
        global email_thread_busy
        try:
            trigger = []
            if visual_conf > 0.0:
                trigger.append("Visual (body language)")
            if audio_score > 0.0:
                trigger.append("Audio (bark/growl)")
            trigger_str = " + ".join(trigger) if trigger else "Combined signal"

            body = f"""
AUTOMATED ALERT — Dog Aggression Detection System

An aggressive dog has been detected.

Duration          : {duration_seconds:.0f} seconds
Triggered by      : {trigger_str}
Visual Confidence : {visual_conf:.2f}
Audio Score       : {audio_score:.2f}
Time              : {time.strftime("%Y-%m-%d %H:%M:%S")}

Please investigate the location immediately.
— Automated Deterrent System
            """.strip()

            msg            = MIMEMultipart()
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = EMAIL_RECEIVER
            msg["Subject"] = EMAIL_SUBJECT
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

            print(f"[Email] ✓ Sent to {EMAIL_RECEIVER}")
        except Exception as e:
            print(f"[Email] ✗ Failed: {e}")
        finally:
            email_thread_busy = False

    threading.Thread(target=_send, daemon=True).start()


# ===================== AUDIO WORKER =====================
def audio_worker(shared_audio_prob, running_flag):
    """
    Separate OS process — completely independent of visual loop speed.
    Pipeline:
      Mic (44100Hz) → resample to 16kHz → YAMNet gate
      → log-mel spectrogram → Audio CNN → Ensemble
      → prob = (cnn_p + ens_p) / 2 → shared memory
    Updates every ~0.9 seconds (3s window, 0.5s slide).
    """
    try:
        print("Loading Audio Engine...")
        cnn_model = tf.keras.models.load_model("audio_cnn_model.h5")
        ensemble = pickle.load(open("audio_ensemble_classifier.pkl", "rb"))
        feat_model = tf.keras.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.get_layer("feature_output").output
        )
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        

        AI_SR          = 16000
        WINDOW_SECONDS = 3      # reduced from 5 → faster updates
        SLIDE_SECONDS  = 0.5    # reduced from 1.0 → more frequent decisions

        dev_info  = sd.query_devices(AUDIO_DEVICE_INDEX, 'input')
        NATIVE_SR = int(dev_info['default_samplerate'])
        audio_queue = Queue()

        def callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())

        class_map_path = tf.keras.utils.get_file(
            "yamnet_map.csv",
            "https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/audioset/yamnet/yamnet_class_map.csv"
        )
        class_names  = np.loadtxt(class_map_path, dtype=str,
                                   delimiter=",", skiprows=1, usecols=2)
        dog_keywords = ["Dog", "Bark", "Growling", "Howl", "Yelp", "Yip"]
        dog_indices  = [i for i, n in enumerate(class_names)
                        if any(k.lower() in n.lower() for k in dog_keywords)]

        with sd.InputStream(device=AUDIO_DEVICE_INDEX, channels=1,
                            samplerate=NATIVE_SR, callback=callback):
            print(f"[Audio] Active — {NATIVE_SR}Hz native → {AI_SR}Hz model input")
            samples = []

            while running_flag.value:
                while not audio_queue.empty():
                    samples.extend(audio_queue.get().flatten())

                if len(samples) >= int(NATIVE_SR * WINDOW_SECONDS):
                    y_raw   = np.array(samples[:int(NATIVE_SR * WINDOW_SECONDS)])
                    samples = samples[int(NATIVE_SR * SLIDE_SECONDS):]

                    y   = librosa.resample(y_raw, orig_sr=NATIVE_SR, target_sr=AI_SR)
                    rms = np.sqrt(np.mean(y ** 2))

                    # Skip silent frames
                    if rms < 0.02:
                        shared_audio_prob.value = 0.0
                        continue

                    y = y / (rms + 1e-6)

                    # YAMNet gate — only proceed if dog sounds detected
                    scores, _, _ = yamnet_model(y)
                    dog_scores   = tf.gather(scores, dog_indices, axis=1).numpy()

                    if np.max(dog_scores) > 0.5:
                        mel     = librosa.feature.melspectrogram(
                                      y=y, sr=AI_SR, n_mels=64)
                        log_mel = librosa.power_to_db(mel, ref=np.max)

                        if log_mel.shape[1] < 157:
                            log_mel = np.pad(
                                log_mel, ((0, 0), (0, 157 - log_mel.shape[1])))
                        else:
                            log_mel = log_mel[:, :157]

                        inp   = np.expand_dims(log_mel, axis=(0, -1))
                        cnn_p = float(cnn_model.predict(inp, verbose=0)[0][0])
                        emb   = feat_model.predict(inp, verbose=0)
                        ens_p = float(ensemble.predict_proba(emb)[0][1])
                        prob  = (cnn_p + ens_p) / 2

                        # Clamp unrealistic spikes
                        if prob > 0.95:
                            prob = 0.85

                        shared_audio_prob.value = prob
                    else:
                        shared_audio_prob.value = 0.0
                else:
                    time.sleep(0.05)

    except Exception as e:
        print(f"[Audio Worker] Crashed: {e}")


# ===================== MAIN =====================
if __name__ == "__main__":

    # Start audio process
    audio_prob = Value('d', 0.0)
    running    = Value('b', True)

    p_audio = Process(target=audio_worker, args=(audio_prob, running))
    p_audio.start()

    # Load visual models
    print("Loading Visual Engine...")
    yolo        = YOLO("yolov8n.pt")
    image_model = tf.keras.models.load_model("dog_aggression_model.h5")
    cap         = cv2.VideoCapture(0)

    # ── Visual state ──
    visual_buffer  = deque(maxlen=VISUAL_BUFFER_LEN)
    visual_confirm = None
    frame_count    = 0
    last_v_score   = 0.0     # holds last CNN result between skipped frames

    # ── Audio state ──
    audio_buffer  = deque(maxlen=AUDIO_BUFFER_LEN)
    audio_confirm = None

    # ── Motion tracking ──
    prev_gray = None

    # ── Status hold ──
    current_status    = "NORMAL"
    last_aggressive_t = 0.0

    print("\nSystem Ready — Press Q to quit")
    print(f"  Motion threshold : {MOTION_THRESHOLD} (lower = more sensitive)")
    print(f"  Calm baseline    : {CALM_BASELINE}")
    print(f"  Visual trigger   : confidence > {VISUAL_CONF_MIN} sustained {VISUAL_CONFIRM_TIME}s")
    print(f"  Audio trigger    : prob > {AUDIO_THRESHOLD} sustained {AUDIO_CONFIRM_TIME}s\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ══════════════════════════════════════════════════
        # STEP 1 — RESIZE FOR SPEED
        # Shrink frame before feeding to YOLO.
        # Bounding boxes scaled back to original for display.
        # ══════════════════════════════════════════════════
        h_orig, w_orig = frame.shape[:2]
        small   = cv2.resize(frame, (480, 360))
        scale_x = w_orig / 480
        scale_y = h_orig / 360

        # ══════════════════════════════════════════════════
        # STEP 2 — MOTION GATE
        # Mean pixel difference between frames.
        # Aggressive dogs move — they lunge, pace, snap.
        # Calm sitting dog → near-zero motion → CNN skipped.
        # ══════════════════════════════════════════════════
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff   = cv2.absdiff(prev_gray, gray)
            motion = float(np.mean(diff))
        else:
            motion = 0.0
        prev_gray  = gray.copy()
        dog_moving = motion > MOTION_THRESHOLD

        # ══════════════════════════════════════════════════
        # STEP 3 — YOLO DETECTION
        # Detect dog (cls=16) on small frame.
        # CNN runs every CNN_SKIP_FRAMES frames only.
        # Between CNN calls, reuse last v_score.
        # ══════════════════════════════════════════════════
        v_score = 0.0
        results = yolo(small, verbose=False, imgsz=224)

        for box in results[0].boxes:
            if int(box.cls[0]) == 16:
                # Scale box back to original frame size
                x1 = int(box.xyxy[0][0] * scale_x)
                y1 = int(box.xyxy[0][1] * scale_y)
                x2 = int(box.xyxy[0][2] * scale_x)
                y2 = int(box.xyxy[0][3] * scale_y)

                crop = frame[max(0, y1):y2, max(0, x1):x2]

                if crop.size > 0:
                    if dog_moving and frame_count % CNN_SKIP_FRAMES == 0:
                        # Moving dog, CNN frame — run classifier
                        img     = cv2.resize(crop, (224, 224)) / 255.0
                        calm_p  = float(
                            image_model.predict(
                                np.expand_dims(img, 0), verbose=0)[0][0])
                        last_v_score = 1.0 - calm_p
                        v_score      = last_v_score

                    elif dog_moving:
                        # Moving dog, skipped CNN frame — reuse last result
                        v_score = last_v_score

                    else:
                        # Stationary dog — always calm regardless of model
                        v_score      = 0.0
                        last_v_score = 0.0

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                break

        # ── Visual smoothing ──
        visual_buffer.append(v_score)
        smoothed_v = float(np.mean(visual_buffer))

        # ── Visual confirm timer ──
        # Score must stay above (calm_baseline + gap) for VISUAL_CONFIRM_TIME
        # seconds continuously before visual_confirmed becomes True
        if smoothed_v > (CALM_BASELINE + 0.08):
            if visual_confirm is None:
                visual_confirm = time.time()
            visual_confirmed = (time.time() - visual_confirm) > VISUAL_CONFIRM_TIME
        else:
            visual_confirm   = None
            visual_confirmed = False

        # ── Visual confidence gap ──
        # How far above calm baseline is the smoothed score?
        # 0.0 at baseline, grows only with genuine aggression signal.
        # Zero until confirm timer passes — spike protection.
        if visual_confirmed:
            visual_confidence = max(
                0.0,
                (smoothed_v - CALM_BASELINE) / (1.0 - CALM_BASELINE)
            )
        else:
            visual_confidence = 0.0

        # ══════════════════════════════════════════════════
        # STEP 4 — AUDIO PIPELINE
        # Reads shared memory from audio process.
        # Smoothed over rolling buffer + confirm timer.
        # Audio being 0.0 (absent/silent) does NOT block
        # visual from triggering on its own.
        # ══════════════════════════════════════════════════
        ap_raw = audio_prob.value
        audio_buffer.append(ap_raw)
        ap = float(np.mean(audio_buffer))

        if ap > AUDIO_THRESHOLD:
            if audio_confirm is None:
                audio_confirm = time.time()
            audio_confirmed = (time.time() - audio_confirm) > AUDIO_CONFIRM_TIME
        else:
            audio_confirm   = None
            audio_confirmed = False

        audio_confidence = ap if audio_confirmed else (ap * 0.5)

        # ══════════════════════════════════════════════════
        # STEP 5 — DECISION
        #
        # Three ways to trigger AGGRESSIVE:
        #
        #  A) Visual alone — confirmed, strong signal
        #     dog is visibly aggressive even if silent
        #
        #  B) Audio alone — confirmed bark/growl
        #     audio pipeline is reliable when signal present
        #
        #  C) Both weak — neither confirmed alone but
        #     both pointing at aggression together
        #
        # Status hold — once AGGRESSIVE, stays for 2s
        # before reverting. Stops flickering.
        # ══════════════════════════════════════════════════
        now = time.time()

        visual_triggers = visual_confirmed and visual_confidence > VISUAL_CONF_MIN
        audio_triggers  = audio_confirmed
        both_weak       = visual_confidence > 0.15 and audio_confidence > 0.25

        if visual_triggers or audio_triggers or both_weak:
            current_status    = "AGGRESSIVE"
            last_aggressive_t = now
        else:
            if (now - last_aggressive_t) > STATUS_HOLD_TIME:
                current_status = "NORMAL"

        status = current_status

        # ══════════════════════════════════════════════════
        # STEP 6 — DETERRENT + EMAIL
        # ══════════════════════════════════════════════════
        if status == "AGGRESSIVE":
            if aggressive_since is None:
                aggressive_since = now

            aggression_duration = now - aggressive_since
            play_deterrent()

            cooldown_ok = (now - email_sent_time) > EMAIL_COOLDOWN_SECONDS
            if (aggression_duration >= EMAIL_TRIGGER_SECONDS
                    and cooldown_ok
                    and not email_thread_busy):
                send_email_alert(aggression_duration, visual_confidence, ap)
                email_sent_time = now
        else:
            aggressive_since = None

        # ══════════════════════════════════════════════════
        # HUD
        # ══════════════════════════════════════════════════
        color = (0, 0, 255) if status == "AGGRESSIVE" else (0, 255, 0)

        # Line 1 — scores
        cv2.putText(frame,
                    f"Vis: {visual_confidence:.2f}  Aud: {ap:.2f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Line 2 — motion
        cv2.putText(frame,
                    f"Motion: {motion:.1f}  {'MOVING' if dog_moving else 'still'}",
                    (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 100), 2)

        # Line 3 — status
        cv2.putText(frame,
                    status,
                    (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Line 4 — deterrent + email countdown (aggressive only)
        if status == "AGGRESSIVE":
            aggression_duration = now - aggressive_since if aggressive_since else 0
            remaining = max(0, EMAIL_TRIGGER_SECONDS - aggression_duration)

            cv2.putText(frame, "ULTRASONIC ON",
                        (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 100, 255), 2)

            mail_text  = f"Mail in {remaining:.0f}s" if remaining > 0 else "MAIL SENT"
            mail_color = (0, 200, 255) if remaining > 0 else (0, 255, 180)
            cv2.putText(frame, mail_text,
                        (20, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mail_color, 2)

        cv2.imshow("Dog Aggression Deterrent", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running.value = False
            break

    cap.release()
    cv2.destroyAllWindows()
    p_audio.join()