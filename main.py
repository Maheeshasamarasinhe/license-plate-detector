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

vehicles = [2, 3, 5, 7]

# Load video
cap = cv2.VideoCapture('./sample.mp4')

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
            
            # Process license plates
            for prediction in license_plate_result['predictions']:
                x = prediction['x']
                y = prediction['y']
                width = prediction['width']
                height = prediction['height']
                confidence = prediction['confidence']
                
                # Calculate bounding box coordinates
                x1 = int(x - width/2)
                y1 = int(y - height/2)
                x2 = int(x + width/2)
                y2 = int(y + height/2)
                
                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car([x1, y1, x2, y2, confidence, 0], track_ids)
                
                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    
                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    
                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    
                    if license_plate_text is not None:
                        # Store results
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': confidence,
                                                                        'text_score': license_plate_text_score}}
                        
                        # Draw license plate bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Display license plate text on TOP of the VEHICLE frame with red background
                        text = license_plate_text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.2
                        font_thickness = 2
                        
                        # Get text size
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                        
                        # Calculate position - top of vehicle bounding box
                        text_x = int(xcar1)
                        text_y = int(ycar1) - 10  # Above the vehicle box
                        
                        # Draw red background rectangle
                        padding = 10
                        bg_x1 = text_x - padding // 2
                        bg_y1 = text_y - text_height - padding
                        bg_x2 = text_x + text_width + padding // 2
                        bg_y2 = text_y + baseline + padding // 2
                        
                        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)  # Red filled rectangle
                        
                        # Draw white text on top of the red background
                        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
                    else:
                        # Draw license plate box even if text not readable
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'License Plate ({confidence:.2f})', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    # Draw license plate box if no car assigned
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