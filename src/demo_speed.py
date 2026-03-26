import cv2
from ultralytics import solutions

# 1. Open the video file
cap = cv2.VideoCapture('traffic1.mp4')

# 2. Define the region (Let's use a straight line across the road again)
line_points = [(0, 400), (1280, 400)] 

# 3. Set up the Speed Estimator
speed_estimator = solutions.SpeedEstimator(
    show=True,                # Show the video window
    model="yolov8n.pt",       # Use our nano AI brain
    region=line_points,       # The line the cars need to cross
    # This is the magic number! It tells the AI that 1 pixel = 0.05 meters.
    # (In the real world, you adjust this based on your camera's height).
    meter_per_pixel=0.05      
)

# 4. The "Brain Loop"
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video finished!")
        break

    # The AI detects, tracks, and calculates speed all at once!
    results = speed_estimator(frame)

# Clean up
cap.release()
cv2.destroyAllWindows()