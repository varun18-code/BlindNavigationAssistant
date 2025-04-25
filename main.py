import cv2
import pyttsx3
import time

# Load the model
net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt",
                               "models/MobileNetSSD_deploy.caffemodel")

# COCO-style labels for the SSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Init camera and TTS
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def get_position(cx, width):
    if cx < width / 3:
        return "left"
    elif cx < 2 * width / 3:
        return "center"
    else:
        return "right"

last_spoken = ""
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
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
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cx = int((x1 + x2) / 2)
            pos = get_position(cx, w)

            objects.append((label, pos))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({pos})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Speak every 3 sec
    if objects and time.time() - last_time > 3:
        spoken = ", ".join(f"{obj} ahead on the {pos}" for obj, pos in objects)
        if spoken != last_spoken:
            engine.say(spoken)
            engine.runAndWait()
            last_spoken = spoken
            last_time = time.time()

    cv2.imshow("Lightweight Blind Navigation Assistant", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
