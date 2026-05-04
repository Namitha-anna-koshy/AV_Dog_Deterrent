from fastapi import FastAPI, Request, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import asyncio
import shutil
import numpy as np
import librosa
from moviepy import VideoFileClip
import time
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sounddevice as sd  # <-- NEW: Live Microphone access

from ai_engine import DogDefenseSystem

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ==========================================
# EMAIL ALERT CONFIGURATION
# ==========================================
SENDER_EMAIL = "saluaneena@gmail.com"      # <-- PUT YOUR SENDING EMAIL HERE
SENDER_PASSWORD = "csuo uwey pndd hukb"    # <-- PUT YOUR 16-DIGIT APP PASSWORD HERE
RECEIVER_EMAIL = "aneenamssalu@gmail.com"    # Target contact

def send_email_async(subject, body):
    """Sends email in a background thread so the video doesn't freeze."""
    def send():
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECEIVER_EMAIL
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"📧 SUCCESS: Email Alert Sent - {subject}")
        except Exception as e:
            print(f"❌ ERROR: Failed to send email. Check credentials. {e}")
    
    threading.Thread(target=send).start()

# ==========================================
# GLOBAL STATE
# ==========================================
system = DogDefenseSystem()
camera = None
video_path = None
audio_full = None
sr = 16000
mode = "LIVE"
current_frame_idx = 0
fps = 30.0
is_running = False  

# --- Timer tracking variables ---
aggression_start_time = None
alert_5s_sent = False
alert_10s_sent = False

# ==========================================
# LIVE MICROPHONE BUFFER
# ==========================================
live_audio_stream = None
live_audio_buffer = np.zeros(16000 * 2, dtype=np.float32) # 2-second rolling memory

def audio_callback(indata, frames, time, status):
    """Constantly updates the 2-second live audio memory"""
    global live_audio_buffer
    try:
        shift = len(indata)
        live_audio_buffer = np.roll(live_audio_buffer, -shift)
        live_audio_buffer[-shift:] = indata[:, 0]
    except:
        pass

def get_camera():
    global camera, video_path, mode
    if mode == "LIVE":
        if camera is None or not camera.isOpened():
            # Use CAP_DSHOW to prevent USB locks on Windows
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    elif mode == "VIDEO":
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(video_path)
    return camera

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global video_path, mode, camera, audio_full, fps, current_frame_idx, is_running
    global aggression_start_time, alert_5s_sent, alert_10s_sent
    global live_audio_stream
    
    # --- STOP LIVE MICROPHONE IF UPLOADING VIDEO ---
    if live_audio_stream is not None:
        live_audio_stream.stop()
        live_audio_stream.close()
        live_audio_stream = None

    system.reset()
    aggression_start_time = None
    alert_5s_sent = False
    alert_10s_sent = False

    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        vc = VideoFileClip(temp_name)
        audio_path = "temp_audio.wav"
        vc.audio.write_audiofile(audio_path, fps=16000, logger=None)
        audio_full, _ = librosa.load(audio_path, sr=16000)
        fps = vc.fps
    except:
        audio_full = np.zeros(16000)
        fps = 30.0

    video_path = temp_name
    mode = "VIDEO"
    current_frame_idx = 0
    is_running = True  
    if camera: camera.release()
    camera = cv2.VideoCapture(video_path)
    
    return {"status": "Video Processed"}

@app.post("/switch_mode")
async def switch_mode(request: Request):
    global mode, camera, audio_full, is_running
    global aggression_start_time, alert_5s_sent, alert_10s_sent
    global live_audio_stream, live_audio_buffer
    
    system.reset()
    aggression_start_time = None
    alert_5s_sent = False
    alert_10s_sent = False
    
    data = await request.json()
    mode = "LIVE"
    audio_full = None 
    is_running = True
    
    # Safely release the camera and let the background thread open it
    if camera is not None: 
        camera.release()
        camera = None

    # --- START LIVE MICROPHONE ---
    try:
        if live_audio_stream is not None:
            live_audio_stream.stop()
            live_audio_stream.close()
        live_audio_buffer = np.zeros(16000 * 2, dtype=np.float32)
        live_audio_stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback)
        live_audio_stream.start()
        print("🎙️ Live Microphone Started")
    except Exception as e:
        print(f"⚠️ Microphone Error: {e}")

    return {"status": f"Switched to {mode}"}

@app.post("/stop_feed")
async def stop_feed():
    global is_running, live_audio_stream
    is_running = False
    system.reset()

    # --- STOP LIVE MICROPHONE ---
    if live_audio_stream is not None:
        live_audio_stream.stop()
        live_audio_stream.close()
        live_audio_stream = None

    return {"status": "Stopped"}

def generate_frames():
    global camera, current_frame_idx, is_running
    global aggression_start_time, alert_5s_sent, alert_10s_sent
    global live_audio_buffer
    
    SKIP_FRAMES = 2 
    frame_counter = 0
    
    while True:
        if not is_running:
            break

        cam = get_camera()
        success, frame = cam.read()
        
        if not success:
            if mode == "VIDEO":
                cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame_idx = 0
                continue
            else:
                break
        
        processed_frame = frame
        
        if frame_counter % (SKIP_FRAMES + 1) == 0:
            chunk = None
            if mode == "VIDEO" and audio_full is not None:
                t_sec = current_frame_idx / fps
                start = int(t_sec * sr)
                end = start + int(2.0 * sr)
                if end < len(audio_full):
                    chunk = audio_full[start:end]
            elif mode == "LIVE":
                # --- GRAB THE LAST 2 SECONDS FROM THE LIVE MIC ---
                chunk = live_audio_buffer.copy()

            processed_frame, data = system.predict(frame, audio_chunk=chunk)
            system.latest_data = data 
            
            # ==========================================
            # ALERT TIMER LOGIC
            # ==========================================
            if data["status"] == "AGGRESSIVE":
                if aggression_start_time is None:
                    aggression_start_time = time.time()
                
                elapsed_time = time.time() - aggression_start_time
                
                if elapsed_time >= 10.0 and not alert_10s_sent:
                    send_email_async(
                        subject="🚨 URGENT: Severe Canine Aggression Detected!",
                        body="Canine Guard System Alert:\n\nA dog has been displaying continuous aggressive behavior for over 10 SECONDS at the monitored location. Immediate intervention may be required."
                    )
                    alert_10s_sent = True
                
                elif elapsed_time >= 5.0 and not alert_5s_sent:
                    send_email_async(
                        subject="⚠️ WARNING: Canine Aggression Alert (5s)",
                        body="Canine Guard System Alert:\n\nAn aggressive dog has been detected for 5 continuous seconds. The ultrasonic deterrent is active."
                    )
                    alert_5s_sent = True
                    
            else:
                aggression_start_time = None
                alert_5s_sent = False
                alert_10s_sent = False
                
        else:
            pass

        frame_counter += 1
        current_frame_idx += 1
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        if is_running and hasattr(system, 'latest_data'):
            await websocket.send_json(system.latest_data)
        else:
            await websocket.send_json({"score": 0, "status": "SAFE"})
        await asyncio.sleep(0.1)