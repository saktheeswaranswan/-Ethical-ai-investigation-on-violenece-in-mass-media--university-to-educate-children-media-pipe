import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load the YOLOv11x-pose model (highest accuracy)
model = YOLO("yolo11x-pose.pt")  # ensure this file is available

# Open video source (replace with 0 for webcam or a file path)
cap = cv2.VideoCapture("bullmike.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Video properties and output setup (HD 1280x720)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = 1280
height = 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/vadighfgffyyggerrfrggggygytttngg.mp4", fourcc, fps, (width, height))

pose_data = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # Resize frame to 1280x720
    frame = cv2.resize(frame, (width, height))

    # Run pose estimation with tracking enabled
    results = model.track(frame, task="pose", persist=True)
    res = results[0]  # result for the current frame

    # Draw skeletons/keypoints on the frame
    annotated_frame = res.plot()

    # If there are detections, extract their data
    if res.boxes is not None and len(res.boxes.xyxy):
        # Get tracking IDs (one per detected person)
        if res.boxes.id is not None:
            track_ids = res.boxes.id.int().cpu().numpy().tolist()
        else:
            track_ids = [None] * len(res.boxes.xyxy)

        # Iterate over each detected person i
        for i, bbox in enumerate(res.boxes.xyxy):
            person_id = int(track_ids[i]) if track_ids[i] is not None else None
            # 17 x [x, y] keypoint coordinates
            keypoints = res.keypoints.xy[i].cpu().numpy().tolist()
            # Bounding box [x_min, y_min, x_max, y_max]
            bbox_xyxy = bbox.cpu().numpy().tolist()

            frame_data = {
                "frame_id": frame_count,
                "person_id": person_id,
                "keypoints": keypoints,
                "bounding_box": bbox_xyxy
            }
            pose_data.append(frame_data)

    # Write the annotated frame to output video
    out.write(annotated_frame)
    frame_count += 1

# Release resources
cap.release()
out.release()

# Save all pose data to JSON file
with open("output/pose_data.json", "w") as f:
    json.dump(pose_data, f, indent=4)

print("✅ Annotated video saved as output/annotated_output.mp4")
print("✅ Pose data saved as output/pottse_data.json")



