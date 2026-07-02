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
from supabase import create_client, Client

# =====================================================================
# 1. HARD-CODED BACKEND CONFIGURATION
# =====================================================================
SUPABASE_URL = "https://ravxgcibqwxnuupcxidt.supabase.co"
# SERVICE_ROLE KEY HERE:
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhdnhnY2licXd4bnV1cGN4aWR0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4MjE5NzYwOCwiZXhwIjoyMDk3NzczNjA4fQ.1CmMwQ34vHN3vKwhylJTpLlaBP9RUctPNESSp_CnCEY"  

FASTAPI_URL = "https://edgevision-gis.onrender.com/api/telemetry/"
API_KEY = "pidec_edge_8f43b2a9e1d7c6f54032b1a8c9d0e7f6"  

# Dynamically force absolute path resolution to eliminate "Can't read ONNX file" errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")

CONF_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45
IMAGE_DIR = os.path.join(BASE_DIR, "captured_potholes")

# Initialize directories and cloud clients
os.makedirs(IMAGE_DIR, exist_ok=True)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================================
# 2. LOCAL SQLITE CACHE INITIALIZATION
# =====================================================================
def init_local_cache():
    db_path = os.path.join(BASE_DIR, "ril_cache.db")
    conn = sqlite3.connect(db_path)
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

def get_gps_coordinates():
    # Fallback default coordinates (Lagos baseline)
    return 6.5244, 3.3792

# =====================================================================
# 3. ASYNCHRONOUS BACKGROUND UPLOADER THREAD (HOTSPOT COMPATIBLE)
# =====================================================================
def network_uploader_worker():
    print("📡 Background Network Worker Loop Activated...")
    db_path = os.path.join(BASE_DIR, "ril_cache.db")
    
    while True:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Read the oldest stored tracking event
        cursor.execute("SELECT id, timestamp, latitude, longitude, pixel_area, image_path FROM telemetry_queue ORDER BY id ASC LIMIT 1")
        row = cursor.fetchone()
        
        if row:
            db_id, timestamp, lat, lon, pixel_area, img_path = row
            filename = os.path.basename(img_path)
            public_url = None
            
            try:
                # Step A: Push local image file to Supabase Storage Bucket
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        supabase_client.storage.from_("pothole-images").upload(
                            file=f, path=filename, file_options={"upsert": "true", "content-type": "image/jpeg"}
                        )
                    public_url = supabase_client.storage.from_("pothole-images").get_public_url(filename)
                
                # Step B: Pass JSON data payload to FastAPI WebGIS
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
                    print(f"🎉 Successfully ingested item {db_id} to WebGIS Backend!")
                    # Success: Clear cache queue record
                    cursor.execute("DELETE FROM telemetry_queue WHERE id = ?", (db_id,))
                    conn.commit()
                    # Delete the file off the Pi's storage to save space
                    if os.path.exists(img_path):
                        os.remove(img_path)
                else:
                    print(f"⚠️ Gateway rejected packet ({response.status_code}). Retrying in 30s...")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"📡 Hotspot Outage/Drop Detected. Data safe in SQLite queue. Retrying in 30s...")
                time.sleep(30)
        else:
            time.sleep(4)
            
        conn.close()

# Fire up network thread as a background worker daemon
uploader_thread = threading.Thread(target=network_uploader_worker, daemon=True)
uploader_thread.start()

# =====================================================================
# 4. OPENCV NATIVE DNN MODEL INITIALIZATION
# =====================================================================
print("🧠 Loading YOLOv11 ONNX via OpenCV DNN Engine...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# Enforce hardware-level performance optimization maps for the ARM architecture
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Wake up the hardware camera node
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

print("🚀 RIL Edge Node Active and Monitoring Camera Feed...")

# =====================================================================
# 5. CORE CAMERA RUNTIME INFERENCE LOOP
# =====================================================================
db_path = os.path.join(BASE_DIR, "ril_cache.db")

try:
    while True:
        frame = picam2.capture_array()
        h_orig, w_orig, _ = frame.shape
        
        # Format raw camera array into a unified 640x640 matrix block
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=False, crop=False)
        net.setInput(blob)
        
        # Execute model forward propagation passes
        outputs = net.forward()
        
        # Shape Correction: Clean batch dimension out if present
        if len(outputs.shape) == 3:
            outputs = outputs[0]
            
        # Shape Correction: Dynamic orientation verification to avoid IndexErrors
        if outputs.shape[0] == 156 or outputs.shape[0] < outputs.shape[1]:
            predictions = outputs.T   # Rotate layout to guarantee an aligned shape (8400, 156)
        else:
            predictions = outputs     # Already sitting in optimal (8400, 156) alignment
        
        num_anchors = predictions.shape[0]  # Exactly 8400 anchors
        
        boxes = []
        confidences = []
        
        # Parse the predictions matrix using explicit row slicing to avoid array boundary leakage
        for i in range(num_anchors):
            row = predictions[i]   # Isolate a single row of length 156 attributes
            scores = row[4:]       # Isolate just the class probabilities
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                # Map scaled ratios cleanly back to the real capture resolution dimensions
                x1 = int((cx - w / 2) * (w_orig / 640.0))
                y1 = int((cy - h / 2) * (h_orig / 640.0))
                width = int(w * (w_orig / 640.0))
                height = int(h * (h_orig / 640.0))
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                
        # Filter overlapping prediction bounding boxes via Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        
        if len(indices) > 0:
            total_pixel_area = 0
            for i in indices.flatten():
                _, _, w_box, h_box = boxes[i]
                total_pixel_area += (w_box * h_box)
                
            # Log exact data acquisition moment
            timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
            lat, lon = get_gps_coordinates()
            
            # Save compressed file locally to handle pipeline handoff
            unique_id = uuid.uuid4().hex
            img_filename = f"pothole_{unique_id}.jpg"
            local_img_path = os.path.join(IMAGE_DIR, img_filename)
            
            # Convert channel mapping back to BGR for OpenCV file-write syntax
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(local_img_path, bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            
            # Atomically drop record details straight into local caching table queue
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO telemetry_queue (timestamp, latitude, longitude, pixel_area, image_path) VALUES (?, ?, ?, ?, ?)",
                (timestamp_str, lat, lon, total_pixel_area, local_img_path)
            )
            conn.commit()
            conn.close()
            
            print(f"💾 Pothole Logged Locally! Area: {total_pixel_area} px. Enqueued for hotspot upload.")
            time.sleep(2.0)  # Standard window delay to prevent logging duplicates of the same road pothole

except KeyboardInterrupt:
    print("\nShutting down RIL Logger System cleanly...")
finally:
    picam2.stop()
