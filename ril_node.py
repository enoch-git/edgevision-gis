import os
import time
import uuid
import sqlite3
import datetime
import threading
import cv2
import numpy as np
import requests
from picamera2 import Picamera2
import onnxruntime as ort
from supabase import create_client, Client

# ==========================================
# 1. HARD-CODED BACKEND CONFIGURATION
# ==========================================
SUPABASE_URL = "https://ravxgcibqwxnuupcxidt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhdnhnY2licXd4bnV1cGN4aWR0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4MjE5NzYwOCwiZXhwIjoyMDk3NzczNjA4fQ.1CmMwQ34vHN3vKwhylJTpLlaBP9RUctPNESSp_CnCEY"

FASTAPI_URL = "https://edgevision-gis.onrender.com/api/telemetry/"
API_KEY = "pidec_edge_8f43b2a9e1d7c6f54032b1a8c9d0e7f6"  # <-- Swap with your active .env secret

MODEL_PATH = "best.onnx"
CONF_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45
IMAGE_DIR = "./captured_potholes"

os.makedirs(IMAGE_DIR, exist_ok=True)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. LOCAL SQLITE CACHE INITIALIZATION
# ==========================================
def init_local_cache():
    conn = sqlite3.connect("ril_cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            pixel_area INTEGER,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

init_local_cache()

# ==========================================
# 3. GNSS/GPS SENSOR INTERFACE PLACEHOLDER
# ==========================================
def get_gps_coordinates():
    """
    Hook your hardware serial/NMEA parsing code here.
    Returns a fallback tuple if GPS has no lock.
    """
    # Fallback default coordinates (e.g., Lagos baseline coordinates)
    fallback_lat = 6.5244
    fallback_lon = 3.3792
    return fallback_lat, fallback_lon

# ==========================================
# 4. ASYNCHRONOUS BACKGROUND UPLOADER THREAD
# ==========================================
def network_uploader_worker():
    print("📡 Background Network Worker Loop Activated...")
    while True:
        conn = sqlite3.connect("ril_cache.db")
        cursor = conn.cursor()
        
        # Pull the oldest item from the cache
        cursor.execute("SELECT id, timestamp, latitude, longitude, pixel_area, image_path FROM telemetry_queue ORDER BY id ASC LIMIT 1")
        row = cursor.fetchone()
        
        if row:
            db_id, timestamp, lat, lon, pixel_area, img_path = row
            filename = os.path.basename(img_path)
            public_url = None
            
            try:
                # Step A: Upload Image to Supabase Bucket
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        supabase_client.storage.from_("pothole-images").upload(
                            file=f, path=filename, file_options={"upsert": "true", "content-type": "image/jpeg"}
                        )
                    public_url = supabase_client.storage.from_("pothole-images").get_public_url(filename)
                
                # Step B: Transmit Payload JSON to FastAPI
                headers = {
                    "X-API-Key": API_KEY,
                    "Content-Type": "application/json"
                }
                payload = {
                    "timestamp": timestamp,
                    "latitude": lat,
                    "longitude": lon,
                    "pothole_pixel_area": pixel_area,
                    "image_url": public_url
                }
                
                response = requests.post(FASTAPI_URL, headers=headers, json=payload, timeout=10)
                
                if response.status_code == 201:
                    print(f"🎉 Successfully ingested item {db_id} to WebGIS Backend! Cleaning disk cache...")
                    # Remove from local database queue
                    cursor.execute("DELETE FROM telemetry_queue WHERE id = ?", (db_id,))
                    conn.commit()
                    # Delete file off SD card to save space on your Pi 3 A+
                    if os.path.exists(img_path):
                        os.remove(img_path)
                else:
                    print(f"⚠️ Gateway rejected packet ({response.status_code}): {response.text}. Retrying in 30s...")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"📡 GSM/Network Outage Detected ({e}). Data stays safe inside Pi local queue. Retrying in 30s...")
                time.sleep(30)
        else:
            # Queue is empty, rest for a bit
            time.sleep(4)
            
        conn.close()

# Start background network daemon thread
uploader_thread = threading.Thread(target=network_uploader_worker, daemon=True)
uploader_thread.start()

# ==========================================
# 5. CORE CAMERA & YOLOv11 ONNX PIPELINE
# ==========================================
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4  # Fully utilize ARMv7 Quad-Core
session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

print("🚀 RIL Edge Intelligence Node Active and Logging...")

try:
    while True:
        frame = picam2.capture_array()
        h_orig, w_orig, _ = frame.shape
        
        # Formatting for YOLOv11 ONNX Matrix [1, 3, 640, 640]
        blob = cv2.resize(frame, (640, 640))
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        
        # Mathematical inference calculation
        outputs = session.run([output_name], {input_name: blob})[0]
        predictions = np.squeeze(outputs).T
        
        boxes = []
        confidences = []
        
        for row in predictions:
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = int((cx - w / 2) * (w_orig / 640.0))
                y1 = int((cy - h / 2) * (h_orig / 640.0))
                width = int(w * (w_orig / 640.0))
                height = int(h * (h_orig / 640.0))
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        
        if len(indices) > 0:
            total_pixel_area = 0
            for i in indices.flatten():
                _, _, w_box, h_box = boxes[i]
                # Calculate bounding box area as pixel area
                total_pixel_area += (w_box * h_box)
                
            # Log the exact moment of logging event
            timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
            lat, lon = get_gps_coordinates()
            
            # Save compressed file locally to handle pipeline handoff
            unique_id = uuid.uuid4().hex
            img_filename = f"pothole_{unique_id}.jpg"
            local_img_path = os.path.join(IMAGE_DIR, img_filename)
            
            # Convert channel mapping back for OpenCV save syntax
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(local_img_path, bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            
            # Lock database link momentarily and drop record into queue
            conn = sqlite3.connect("ril_cache.db")
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO telemetry_queue (timestamp, latitude, longitude, pixel_area, image_path) VALUES (?, ?, ?, ?, ?)",
                (timestamp_str, lat, lon, total_pixel_area, local_img_path)
            )
            conn.commit()
            conn.close()
            
            print(f"💾 Pothole Logged Locally! Area: {total_pixel_area} px. Enqueued for transmission.")
            
            # Brief cooldown to prevent spamming duplicate frames of the same pothole
            time.sleep(2.0)

except KeyboardInterrupt:
    print("\nShutting down RIL Logger System cleanly...")
finally:
    picam2.stop()