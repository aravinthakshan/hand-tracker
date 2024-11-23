import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# Initialize SAM
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# For click-based segmentation
def mouse_click(event, x, y, flags, param):
    global input_point, input_label
    if event == cv2.EVENT_LBUTTONDOWN:
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        
cv2.namedWindow('SAM Segmentation')
cv2.setMouseCallback('SAM Segmentation', mouse_click)

input_point = None
input_label = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    predictor.set_image(frame)
    
    if input_point is not None:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        mask = masks[np.argmax(scores)]
        
        frame_with_mask = frame.copy()
        frame_with_mask[mask] = frame_with_mask[mask] * [0.7, 0.3, 0.3]  #
        
        cv2.circle(frame_with_mask, (input_point[0][0], input_point[0][1]), 5, (0, 255, 0), -1)
        
        cv2.imshow('SAM Segmentation', frame_with_mask)
    else:
        cv2.imshow('SAM Segmentation', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):  
        input_point = None
        input_label = None

cap.release()
cv2.destroyAllWindows()
