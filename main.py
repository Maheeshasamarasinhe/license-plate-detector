from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')

# Initialize Roboflow Inference Client for license plate detection
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="fcCYBiPPncENLdquyj7F"
)

cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output.mp4', fourcc, fps, (frame_width, frame_height))

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                # Draw vehicle bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Vehicle', (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates using Roboflow API
        try:
            license_plate_result = CLIENT.infer(frame, model_id="license-plate-recognition-rxg4e/4")
            
            # Draw license plates
            for prediction in license_plate_result['predictions']:
                x = prediction['x']
                y = prediction['y']
                width = prediction['width']
                height = prediction['height']
                confidence = prediction['confidence']
                
                # Draw license plate bounding box
                x1 = int(x - width/2)
                y1 = int(y - height/2)
                x2 = int(x + width/2)
                y2 = int(y + height/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'License Plate ({confidence:.2f})', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"License plate detection error on frame {frame_nmr}: {e}")

        # Write frame to output video
        out.write(frame)
        print(f'Processing frame {frame_nmr}')

cap.release()
out.release()

print("Video saved as output.mp4")

# write results
write_csv(results, './test.csv')