# ============================================================
# Dog Aggression Deterrent System — Final Demo Version
# Features:
#   - Real-time dog detection (YOLOv8 + CNN + YAMNet)
#   - Audible deterrent sound (replaces ultrasonic for demo)
#   - Email alert to animal control after sustained aggression
#   - Dynamic location via headless browser (Selenium)
#     using the browser's native navigator.geolocation — same
#     Wi-Fi positioning engine that Google Maps uses, zero API key needed.
#     Falls back to location_config.json, then hardcoded coords.
#   - Temporal Smoothing via Sliding Window + Grace Period
#     to prevent brief non-aggressive frames from resetting the
#     aggression timer, ensuring reliable email alert triggering.
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
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from multiprocessing import Process, Value, Queue
from ultralytics import YOLO
from collections import deque
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# ===================== EMAIL CONFIG =====================
EMAIL_SENDER    = ""                                           #Enter sender's email address here
EMAIL_PASSWORD  = ""                                           #Enter sender's email password here
EMAIL_RECEIVER  = ""                                           #Enter the recipient's email address here
EMAIL_SUBJECT   = "⚠️ Aggressive Dog Detected — Immediate Attention Required"

EMAIL_TRIGGER_SECONDS  = 5
EMAIL_COOLDOWN_SECONDS = 60

# ===================== DETERRENT SOUND CONFIG =====================
DETERRENT_FREQ     = 2000
DETERRENT_VOLUME   = 0.6
DETERRENT_SR       = 44100
DETERRENT_DURATION = 1.0

# ===================== LOCATION CONFIG =====================
#Enter details of the location where the system is deployed. This is used for email alerts.
FALLBACK_LOCATION    = "SJCET Campus, Palai, Kottayam, Kerala"
FALLBACK_COORDINATES = "9.7270,76.7108"
LOCATION_REFRESH_INTERVAL = 300
SELENIUM_GEO_TIMEOUT = 20

# ===================== TEMPORAL SMOOTHING CONFIG =====================
# Sliding window: number of recent frames used to compute majority vote.
# Higher = more stable but slightly slower to react. Recommended: 10–20.
SMOOTHING_WINDOW_SIZE = 15

# Aggression ratio threshold within the sliding window.
# E.g. 0.4 means: if 40% or more of recent frames are AGGRESSIVE → treat as AGGRESSIVE.
# Lower = more sensitive; higher = stricter. Recommended: 0.4–0.6.
AGGRESSION_RATIO_THRESHOLD = 0.4

# Grace period (seconds): brief non-aggressive detections within this
# window do NOT reset the aggression timer.
# E.g. 2.0 → up to 2 seconds of "quiet" is tolerated mid-aggression.
GRACE_PERIOD_SECONDS = 2.0


# ===================== STATE FLAGS =====================
sound_playing     = False
sound_lock        = threading.Lock()
email_sent_time    = 0.0
email_display_time = 0.0   # tracks when to show/hide the "EMAIL SENT" banner
aggressive_since   = None
email_thread_busy  = False

# Grace period: timestamp when the system last saw an AGGRESSIVE frame.
# Used to decide if a non-aggressive gap is still within tolerance.
last_aggressive_time = None

# Sliding window: stores per-frame binary aggression labels (1 = aggressive, 0 = normal).
frame_aggression_window = deque(maxlen=SMOOTHING_WINDOW_SIZE)

# Shared location state (updated by background thread)
_location_lock       = threading.Lock()
_current_location    = FALLBACK_LOCATION
_current_coordinates = FALLBACK_COORDINATES
_location_source     = "fallback"   # "browser" | "config" | "fallback"


# ============================================================
#  LOCATION SUBSYSTEM  —  Selenium headless browser
# ============================================================

CONFIG_FILE = Path("location_config.json")


def _load_config_location():
    """
    Reads location_config.json.
    Returns (location_str, coord_str) or None if blank/absent.
    Auto-creates a template file on first run.
    """
    if not CONFIG_FILE.exists():
        template = {
            "_comment": "Fill 'location' and 'coordinates' to hard-override the browser fix.",
            "location": "",
            "coordinates": ""
        }
        CONFIG_FILE.write_text(json.dumps(template, indent=2))
        print(f"[Location] Created {CONFIG_FILE} — leave blank to use browser geolocation.")
        return None
    try:
        data  = json.loads(CONFIG_FILE.read_text())
        loc   = data.get("location",    "").strip()
        coord = data.get("coordinates", "").strip()
        if loc and coord:
            return loc, coord
    except Exception as e:
        print(f"[Location] Config read error: {e}")
    return None


def _nominatim_reverse(lat, lng):
    """
    Free reverse-geocoding via OpenStreetMap Nominatim.
    Returns a readable address string, or coordinate string on failure.
    """
    try:
        import urllib.request
        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lng}&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "DogAggressionSystem/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return data.get("display_name", f"{lat:.6f},{lng:.6f}")
    except Exception as e:
        print(f"[Location] Nominatim error: {e}")
        return f"{lat:.6f},{lng:.6f}"


_GEO_HTML = """<!DOCTYPE html><html><body><script>
window._geoResult = null;
navigator.geolocation.getCurrentPosition(
    function(pos) {
        window._geoResult = {
            lat: pos.coords.latitude,
            lng: pos.coords.longitude,
            acc: pos.coords.accuracy
        };
    },
    function(err) {
        window._geoResult = {error: err.message};
    },
    {enableHighAccuracy: false, timeout: 15000, maximumAge: 60000}
);
</script></body></html>"""


def _start_geo_server():
    """
    Spins up a minimal HTTP server on a free localhost port in a
    background thread. Returns (server, port).
    """
    import socketserver
    import http.server

    html_bytes = _GEO_HTML.encode()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, fmt, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port   = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _fetch_browser_location():
    """
    Opens headless Chrome, calls navigator.geolocation, reverse-geocodes.
    Returns (location_str, coord_str) or None on failure.
    """
    server = None
    driver = None
    try:
        server, port = _start_geo_server()
        url = f"http://127.0.0.1:{port}/"

        opts = ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--log-level=3")
        opts.add_argument("--silent")
        opts.add_experimental_option("prefs", {
            "profile.default_content_setting_values.geolocation": 1
        })

        service = ChromeService(
            ChromeDriverManager().install(),
            log_output=os.devnull
        )
        driver = webdriver.Chrome(service=service, options=opts)
        driver.get(url)

        print(f"[Location] Browser open — waiting up to {SELENIUM_GEO_TIMEOUT}s for fix...")

        deadline = time.time() + SELENIUM_GEO_TIMEOUT
        result   = None
        while time.time() < deadline:
            result = driver.execute_script("return window._geoResult;")
            if result is not None:
                break
            time.sleep(0.5)

        if result is None:
            print("[Location] Browser timed out waiting for geolocation.")
            return None

        if "error" in result:
            print(f"[Location] Browser geolocation error: {result['error']}")
            return None

        lat = result["lat"]
        lng = result["lng"]
        acc = result.get("acc", 0)

        coord_str = f"{lat:.6f},{lng:.6f}"
        loc_str   = _nominatim_reverse(lat, lng)

        print(f"[Location] ✅ Browser fix: {loc_str}  (±{acc:.0f}m)")
        return loc_str, coord_str

    except Exception as e:
        print(f"[Location] Selenium error: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        if server:
            try:
                server.shutdown()
            except Exception:
                pass


def _update_location():
    """
    Priority:
      1. location_config.json  (manual override — highest priority)
      2. Headless browser via navigator.geolocation
      3. Hardcoded fallback constants
    """
    global _current_location, _current_coordinates, _location_source

    cfg = _load_config_location()
    if cfg:
        with _location_lock:
            _current_location    = cfg[0]
            _current_coordinates = cfg[1]
            _location_source     = "config"
        print(f"[Location] Using config override: {cfg[0]}")
        return

    result = _fetch_browser_location()
    if result:
        with _location_lock:
            _current_location    = result[0]
            _current_coordinates = result[1]
            _location_source     = "browser"
        return

    with _location_lock:
        _current_location    = FALLBACK_LOCATION
        _current_coordinates = FALLBACK_COORDINATES
        _location_source     = "fallback"
    print(f"[Location] ⚠️  Using hardcoded fallback: {FALLBACK_LOCATION}")


def _location_refresh_loop():
    """Background thread: fetches on startup then refreshes periodically."""
    _update_location()
    if LOCATION_REFRESH_INTERVAL > 0:
        while True:
            time.sleep(LOCATION_REFRESH_INTERVAL)
            _update_location()


def get_location():
    """Returns (location_str, coord_str, source_str) thread-safely."""
    with _location_lock:
        return _current_location, _current_coordinates, _location_source


# ===================== DETERRENT SOUND =====================
def play_deterrent():
    global sound_playing
    with sound_lock:
        if sound_playing:
            return
        sound_playing = True

    def _play():
        global sound_playing
        try:
            t    = np.linspace(0, DETERRENT_DURATION, int(DETERRENT_SR * DETERRENT_DURATION))
            wave = DETERRENT_VOLUME * np.sin(2 * np.pi * DETERRENT_FREQ * t).astype(np.float32)
            sd.play(wave, samplerate=DETERRENT_SR)
            sd.wait()
        except Exception as e:
            print(f"[Sound] Error: {e}")
        finally:
            sound_playing = False

    threading.Thread(target=_play, daemon=True).start()


# ===================== EMAIL ALERT =====================
def send_email_alert(duration_seconds, v_score, ap):
    global email_thread_busy
    email_thread_busy = True

    def _send():
        global email_thread_busy
        try:
            loc, coord, source = get_location()
            source_label = {
                "browser":  "Headless Browser (navigator.geolocation)",
                "config":   "Manual Config File",
                "fallback": "Hardcoded Fallback"
            }.get(source, source)

            body = f"""
AUTOMATED ALERT — Dog Aggression Detection System

An aggressive dog has been detected continuously for {duration_seconds:.0f} seconds.

Detection Details:
  Visual Aggression Score : {v_score:.2f}
  Audio Aggression Score  : {ap:.2f}
  Duration                : {duration_seconds:.0f} seconds
  Time                    : {time.strftime("%Y-%m-%d %H:%M:%S")}

Location Details:
  Address      : {loc}
  Coordinates  : {coord}
  Google Maps  : https://maps.google.com/?q={coord}
  Source       : {source_label}

This alert was generated automatically by the Dog Aggression Detection System.
Please investigate the location immediately.

— Automated Deterrent System
            """.strip()

            msg = MIMEMultipart()
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = EMAIL_RECEIVER
            msg["Subject"] = EMAIL_SUBJECT
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

            print(f"[Email] ✅ Alert sent to {EMAIL_RECEIVER}")

        except Exception as e:
            print(f"[Email] ❌ Failed: {e}")
        finally:
            email_thread_busy = False

    threading.Thread(target=_send, daemon=True).start()


# ===================== AUDIO WORKER PROCESS =====================
def audio_worker(shared_audio_prob, running_flag):

    print("Loading Audio AI Engine...")
    cnn_model = tf.keras.models.load_model("models/audio_cnn_model.h5")
    ensemble  = pickle.load(open("models/audio_ensemble_classifier.pkl", "rb"))

    feature_model = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("feature_output").output
    )

    yamnet_model = hub.load("models/tfhub_cache/9616fd04ec2360621642ef9455b84f4b668e219e")

    DEVICE_INDEX   = 9
    AI_SR          = 16000
    WINDOW_SECONDS = 5

    dev_info  = sd.query_devices(DEVICE_INDEX, 'input')
    NATIVE_SR = int(dev_info['default_samplerate'])

    audio_queue = Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    class_map_path = tf.keras.utils.get_file(
        "yamnet_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    )

    class_names = np.loadtxt(
        class_map_path, dtype=str, delimiter=",", skiprows=1, usecols=2
    )

    dog_keywords = ["Dog", "Bark", "Growling", "Howl", "Yelp", "Yip"]
    dog_indices  = [
        i for i, name in enumerate(class_names)
        if any(k.lower() in name.lower() for k in dog_keywords)
    ]

    with sd.InputStream(device=DEVICE_INDEX,
                        channels=1,
                        samplerate=NATIVE_SR,
                        callback=callback):

        print(f"Audio Hardware Active: {NATIVE_SR}Hz")
        samples = []

        while running_flag.value:
            while not audio_queue.empty():
                samples.extend(audio_queue.get().flatten())

            if len(samples) >= (NATIVE_SR * WINDOW_SECONDS):
                y_raw   = np.array(samples[:NATIVE_SR * WINDOW_SECONDS])
                samples = samples[int(NATIVE_SR * 1.0):]

                y   = librosa.resample(y_raw, orig_sr=NATIVE_SR, target_sr=AI_SR)
                rms = np.sqrt(np.mean(y ** 2))

                if rms < 0.02:
                    shared_audio_prob.value = 0.0
                    continue

                y = y / (rms + 1e-6)

                scores, _, _ = yamnet_model(y)
                dog_scores   = tf.gather(scores, dog_indices, axis=1).numpy()

                if np.max(dog_scores) > 0.5:
                    mel     = librosa.feature.melspectrogram(y=y, sr=AI_SR, n_mels=64)
                    log_mel = librosa.power_to_db(mel, ref=np.max)

                    if log_mel.shape[1] < 157:
                        log_mel = np.pad(log_mel, ((0, 0), (0, 157 - log_mel.shape[1])))
                    else:
                        log_mel = log_mel[:, :157]

                    inp   = np.expand_dims(log_mel, axis=(0, -1))
                    cnn_p = float(cnn_model.predict(inp, verbose=0)[0][0])
                    emb   = feature_model.predict(inp, verbose=0)
                    ens_p = float(ensemble.predict_proba(emb)[0][1])
                    prob  = (cnn_p + ens_p) / 2

                    if prob > 0.95:
                        prob = 0.85

                    shared_audio_prob.value = prob
                else:
                    shared_audio_prob.value = 0.0
            else:
                time.sleep(0.05)


# ============================================================
#  TEMPORAL SMOOTHING HELPERS
# ============================================================

def compute_smoothed_aggression(raw_frame_label: int) -> bool:
    """
    Pushes the raw per-frame label (1 = aggressive, 0 = normal) into
    the sliding window and returns True if the smoothed window majority
    exceeds AGGRESSION_RATIO_THRESHOLD.

    Args:
        raw_frame_label: 1 if this frame's raw signals indicate aggression,
                         0 otherwise.

    Returns:
        True  → system considers the current moment as AGGRESSIVE (smoothed)
        False → system considers the current moment as NORMAL (smoothed)
    """
    frame_aggression_window.append(raw_frame_label)

    if len(frame_aggression_window) == 0:
        return False

    aggression_ratio = sum(frame_aggression_window) / len(frame_aggression_window)
    return aggression_ratio >= AGGRESSION_RATIO_THRESHOLD


def apply_grace_period(smoothed_aggressive: bool, now: float) -> bool:
    """
    Applies the grace period logic on top of the smoothed label.

    If smoothed_aggressive is False but the last confirmed aggression was
    within GRACE_PERIOD_SECONDS ago, we treat the current frame as still
    AGGRESSIVE (i.e., we tolerate the brief dip).

    Args:
        smoothed_aggressive: output of compute_smoothed_aggression()
        now: current timestamp (time.time())

    Returns:
        True  → treat as AGGRESSIVE (either genuinely or within grace)
        False → treat as NORMAL (gap exceeded grace period)
    """
    global last_aggressive_time

    if smoothed_aggressive:
        # Update the last-seen-aggressive timestamp
        last_aggressive_time = now
        return True

    # Not currently aggressive — check if we're still within the grace window
    if last_aggressive_time is not None:
        gap = now - last_aggressive_time
        if gap <= GRACE_PERIOD_SECONDS:
            # Within grace period: suppress the reset
            return True

    return False


# ===================== MAIN =====================
if __name__ == "__main__":

    AUDIO_THRESHOLD = 0.7
    CONFIRM_TIME    = 1.5

    # Start location refresh in background (non-blocking)
    loc_thread = threading.Thread(target=_location_refresh_loop, daemon=True)
    loc_thread.start()

    audio_prob = Value('d', 0.0)
    running    = Value('b', True)

    p_audio = Process(target=audio_worker, args=(audio_prob, running))
    p_audio.start()

    print("Loading Visual Engines...")
    yolo        = YOLO("yolov8n.pt")
    image_model = tf.keras.models.load_model("models/dog_aggression_model.h5")

    cap = cv2.VideoCapture(0)

    audio_buffer  = deque(maxlen=6)
    confirm_start = None

    print("System Ready.")
    print(f"  Deterrent tone      : {DETERRENT_FREQ}Hz (demo mode)")
    print(f"  Email trigger       : after {EMAIL_TRIGGER_SECONDS}s of aggression")
    print(f"  Email cooldown      : {EMAIL_COOLDOWN_SECONDS}s between alerts")
    print(f"  Location refresh    : every {LOCATION_REFRESH_INTERVAL}s")
    print(f"  Smoothing window    : {SMOOTHING_WINDOW_SIZE} frames "
          f"(threshold: {AGGRESSION_RATIO_THRESHOLD:.0%})")
    print(f"  Grace period        : {GRACE_PERIOD_SECONDS}s tolerance on brief dips")
    print("  Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Visual inference ──────────────────────────────────────
        v_score = 0.0
        results = yolo(frame, verbose=False, imgsz=320)

        for box in results[0].boxes:
            if int(box.cls[0]) == 16:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    img    = cv2.resize(crop, (224, 224)) / 255.0
                    calm_p = float(
                        image_model.predict(np.expand_dims(img, 0), verbose=0)[0][0]
                    )
                    v_score = 1.0 - calm_p
                break

        # ── 2. Audio inference ───────────────────────────────────────
        ap_raw = audio_prob.value
        audio_buffer.append(ap_raw)
        ap = np.mean(audio_buffer)

        if ap > AUDIO_THRESHOLD:
            if confirm_start is None:
                confirm_start = time.time()
            audio_aggressive = (time.time() - confirm_start) > CONFIRM_TIME
        else:
            confirm_start    = None
            audio_aggressive = False

        # ── 3. Raw per-frame aggression label ────────────────────────
        # 1 if either modality fires, 0 otherwise.
        raw_label = 1 if (v_score > 0.45 or audio_aggressive) else 0

        # ── 4. Temporal smoothing (sliding window majority vote) ──────
        smoothed_aggressive = compute_smoothed_aggression(raw_label)

        # ── 5. Grace period (suppress brief non-aggressive gaps) ──────
        now = time.time()
        final_aggressive = apply_grace_period(smoothed_aggressive, now)

        status = "AGGRESSIVE" if final_aggressive else "NORMAL"

        # ── 6. Aggression timer + email trigger ───────────────────────
        if status == "AGGRESSIVE":

            if aggressive_since is None:
                aggressive_since = now

            aggression_duration = now - aggressive_since

            play_deterrent()

            # Send email once per aggression episode (email_sent_time resets when dog calms down)
            if (aggression_duration >= EMAIL_TRIGGER_SECONDS
                    and email_sent_time == 0.0
                    and not email_thread_busy):
                send_email_alert(aggression_duration, v_score, ap)
                email_sent_time    = now
                email_display_time = now

        else:
            # Dog calmed down — reset everything so next aggression triggers fresh email + banner
            aggressive_since   = None
            email_sent_time    = 0.0
            email_display_time = 0.0

        # ── 7. HUD ────────────────────────────────────────────────────
        color = (0, 0, 255) if status == "AGGRESSIVE" else (0, 255, 0)

        # Line 1: Visual and Audio scores
        cv2.putText(frame,
                    f"Visual: {v_score:.2f}  |  Audio: {ap:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Line 2: AGGRESSIVE / NORMAL status
        cv2.putText(frame,
                    status,
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Line 3: EMAIL SENT banner — appears the moment email fires, clears after 5 seconds
        if email_display_time > 0 and (now - email_display_time) < 5.0:
            cv2.putText(frame,
                        "EMAIL SENT",
                        (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

        cv2.imshow("Dog Aggression Deterrent System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running.value = False
            break

    cap.release()
    cv2.destroyAllWindows()
    p_audio.join()