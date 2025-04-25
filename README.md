# 🦯 Blind Navigation Assistant

A real-time object detection and voice-guided navigation system to assist visually impaired users, built using Python, OpenCV, and Pyttsx3.

---

## 🔍 Overview

The Blind Navigation Assistant is a vision-based application that detects everyday objects from a webcam feed using a pre-trained deep learning model and provides **directional voice feedback** to help visually impaired users navigate safely.

The system processes video frames, identifies relevant objects using **MobileNet SSD**, and uses **text-to-speech** to announce both the object and its spatial location (left, center, right). It also includes **collision warnings**, **night mode**, **smart filtering**, and automatic **snapshot logging**.

---

## 🎯 Features

✅ Real-time object detection via webcam  
✅ Direction-aware voice alerts using Pyttsx3  
✅ Collision warning for nearby obstacles  
✅ Smart object filtering (ignores non-navigational objects)  
✅ Night mode: Boosts brightness in low-light conditions  
✅ Snapshot saving of close objects  
✅ Auto-generated activity logs  
✅ Lightweight and beginner-friendly codebase

---

## 📁 Project Structure

BlindNavigationAssistant/
├── models/                            # Pre-trained model files
│   ├── MobileNetSSD_deploy.caffemodel
│   └── MobileNetSSD_deploy.prototxt
├── snapshots/                         # Directory for saved snapshots
├── main.py                            # Main application script
├── detection_log.txt                  # Log file for detected objects
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── LICENSE                            # Project license

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/BlindNavigationAssistant.git
cd BlindNavigationAssistant
2. Install Dependencies
Create a virtual environment (recommended), then install:

bash

pip install -r requirements.txt

3. Download the Model Files
Place the following in the models/ folder:

MobileNetSSD_deploy.prototxt

MobileNetSSD_deploy.caffemodel

4. Run the Application
bash
python main.py

🧠 How It Works
The video feed is split into left, center, and right zones.

Detected objects are mapped to their position and announced (e.g., "Person ahead on the left").

If an object is too close, a collision alert is issued, a snapshot is saved, and the event is logged.

In low-light conditions, the system boosts brightness (Night Mode).

📈 Future Enhancements
✅ GUI with Tkinter or PyQt

✅ Raspberry Pi deployment for wearable prototypes

✅ Android app using Kivy or Flutter + TFLite

✅ YOLOv8 upgrade for faster & more accurate detection

✅ Distance estimation using stereo vision or ultrasonic sensors

🧪 Requirements
Python 3.7+

OpenCV

Numpy

Pyttsx3

Install with:

bash
pip install opencv-python pyttsx3 numpy

📝 License
This project is licensed under the MIT License.

🙌 Acknowledgements
MobileNet SSD

OpenCV team

Python & Pyttsx3 community

👤 Author
Varun Pingale
