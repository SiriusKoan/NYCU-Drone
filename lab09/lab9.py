import numpy as np
from numpy import random
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box

# Load YOLOv7-tiny model
WEIGHT = './runs/train/yolov7-lab093/weights/best.pt'  # Change this to the path of your trained
device = "cuda" if torch.cuda.is_available() else "cpu"
model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Open video file
input_video_path = './lab09_test.mp4'  # Change this to the path of your input video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to save the output
output_video_path = 'output_video.mp4'  # Change this to the desired output video file path
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, image = cap.read()
    if not ret:
        break

    image_orig = image.copy()
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    
    if device == "cuda":
        image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
    else:
        image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)

    with torch.no_grad():
        output = model(image)[0]
    
    output = non_max_suppression_kpt(output, conf_thres=0.25, iou_thres=0.65)[0]
    
    # Draw label and confidence on the image
    output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
    for *xyxy, conf, cls in output:
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image_orig, label=label, color=colors[int(cls)], line_thickness=1)

    # Write the frame with detections to the output video file
    out.write(image_orig)

    cv2.imshow("Detected", image_orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
