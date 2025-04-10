# extract_frames.py
import cv2
import os

os.makedirs("frames", exist_ok=True)

cap = cv2.VideoCapture("video.mp4")
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"frames/frame_{i:04d}.png", frame)
    i += 1

cap.release()
print(f"✅ {i} frames extraites dans le dossier 'frames'")
