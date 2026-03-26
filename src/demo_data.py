import cv2
import csv
from ultralytics import solutions

# --- NEW PATHS ---
# The "../" tells Python to go out of the 'src' folder to find these!
video_path = '../test_video/traffic1.mp4'
output_csv = '../output/traffic_report.csv'

# 1. Create and open the CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame Number", "Total Vehicles Counted"]) # Header row

    # 2. Set up the video and AI
    cap = cv2.VideoCapture(video_path)
    
    # Safety Check: Did it actually find the video?
    if not cap.isOpened():
        print(f"❌ ERROR: Could not find the video at {video_path}")
        exit()

    line_points = [(0, 400), (1280, 400)]
    counter = solutions.ObjectCounter(show=True, region=line_points, model="yolov8n.pt")

    frame_count = 0

    # 3. The Brain Loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("✅ Video finished naturally.")
            break
            
        frame_count += 1
        results = counter(frame)
        
        # 4. Save to spreadsheet
        current_count = counter.in_count
        writer.writerow([frame_count, current_count]) 
        
        # 5. Print it to the terminal so we can see it working!
        print(f"Writing Data -> Frame: {frame_count} | Total Cars: {current_count}")

        # 6. Stop safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 You pressed 'q'. Stopping safely...")
            break

print(f"📁 All data successfully saved to {output_csv}!")
cap.release()
cv2.destroyAllWindows()