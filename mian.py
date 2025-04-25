import cv2
import pyttsx3
import time
import os

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt",
                               "models/MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

essential_objects = ["person", "car", "bottle", "chair", "bus", "train", "sofa"]

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

cap = cv2.VideoCapture(0)
last_spoken = ""
last_time = 0

# Ensure snapshot folder exists
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

def get_position(cx, width):
    if cx < width / 3:
        return "left"
    elif cx < 2 * width / 3:
        return "center"
    else:
        return "right"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 📷 Night mode: auto brightness boost
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 60:
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    objects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label not in essential_objects:
                continue  # Filter to only useful classes

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cx = int((x1 + x2) / 2)
            pos = get_position(cx, w)

            objects.append((label, pos))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({pos})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 🛑 Collision warning & snapshot
            if (y2 - y1) > h * 0.5:
                engine.say(f"Warning! {label} is very close")
                engine.runAndWait()
                snapshot_path = f"snapshots/{label}_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_path, frame)

    # 🔊 Speak summary every 3 seconds
    if objects and time.time() - last_time > 3:
        spoken = ", ".join(f"{obj} ahead on the {pos}" for obj, pos in objects)
        if spoken != last_spoken:
            engine.say(spoken)
            engine.runAndWait()
            last_spoken = spoken
            last_time = time.time()

            # 📝 Log to file
            with open("detection_log.txt", "a") as log:
                log.write(f"{time.ctime()} - {spoken}\n")

    cv2.imshow("Blind Navigation Assistant", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
