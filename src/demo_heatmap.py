import cv2
from ultralytics import solutions

# 1. Open the video
cap = cv2.VideoCapture('traffic1.mp4')

# 2. Set up the Heatmap tool
heatmap = solutions.Heatmap(
    show=True,                  # Show the video window
    model="yolov8n.pt",         # Use the AI brain
    colormap=cv2.COLORMAP_JET   # Use a cool Red/Yellow/Blue color scheme
)

# 3. The Brain Loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # The AI generates a thermal map of where the cars spend the most time!
    frame = heatmap(frame)

cap.release()
cv2.destroyAllWindows()