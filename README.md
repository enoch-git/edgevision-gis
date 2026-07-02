# RIL-Edge: Road Infrastructure Logger (Edge AI Node)

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-DNN--Engine-green.svg)](https://opencv.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%203%20A%2B-red.svg)](https://www.raspberrypi.com/)
[![Database](https://img.shields.io/badge/Cache-SQLite3-orange.svg)](https://www.sqlite.org/)

RIL-Edge is the localized computer vision intelligence component of the **Road Infrastructure Logger (RIL)** system. Deployed on a highly resource-constrained Raspberry Pi 3 A+ (512MB RAM), this edge node captures real-time video streams from a vehicle-mounted camera, executes real-time object detection to identify potholes using a highly optimized **YOLOv11 ONNX** model, tracks defect sizing, and uploads synchronized GIS telemetry data to a FastAPI WebGIS gateway and Supabase cloud cluster.

---

## 🚀 Key Features

* **Edge AI on Constrained Silicon:** Runs a custom-trained YOLOv11 object detection model smoothly on 512MB of system RAM by swapping out heavy framework runtimes for native C++ optimized **OpenCV DNN** inference execution.
* **Vectorized Post-Processing:** Replaced heavy iterative Python loop parsing with high-speed NumPy vectorized matrix slicing and axis evaluation, bypassing array boundary leaks and accelerating processing frame rates.
* **Fault-Tolerant Multi-Threaded Architecture:** * **Main Thread:** Handles real-time hardware frame capture and object profiling. Logs events into a local SQLite queue in milliseconds.
  * **Background Daemon Thread:** Monitors the local SQLite queue, checking for active phone hotspot/network connectivity to upload image blobs to Supabase Storage and transmit telemetry JSON payloads to FastAPI.
* **Zero-Drop Data Sync:** Telemetry timestamps are cached at the exact microsecond of local hardware detection, ensuring GIS coordinates and analytical timelines remain pixel-perfect even during extended cellular network outages.

---

## 🛠️ Hardware Stack

* **Processing Unit:** Raspberry Pi 3 Model A+ (Broadcom BCM2837B0, Quad-core Cortex-A53 @ 1.4GHz, 512MB LPDDR2 SDRAM).
* **Imaging Sensor:** Raspberry Pi Camera Module V1 (OmniVision OV5647 / 5-megapixel sensor) or equivalent CSI-connected lens.
* **Connectivity Setup:** Wi-Fi connection bridged via a mobile/personal hotspot configuration.

---

## 📦 System Dependency Architecture

Because cutting-edge execution layers like Python 3.13 do not always have precompiled binaries uploaded for older ARM core configurations, the project environment is designed to inherit system-optimized pre-compilations maintained directly by the Linux kernel repository.
