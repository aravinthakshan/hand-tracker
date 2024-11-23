import cv2
from ultralytics import YOLO
import numpy as np
import torch
import time
from segment_anything import sam_model_registry, SamPredictor

# Initialize YOLO
model = YOLO('yolov8n.pt')  

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    
    annotated_frame = results[0].plot()
    
    # Display FPS
    fps = 1 / (time.time() - results[0].speed['preprocess'] + 1e-9)
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
