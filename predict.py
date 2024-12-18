import os
from ultralytics import YOLO
import cv2

# Load the trained model
model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')
model = YOLO(model_path)

# Set detection threshold
threshold = 0.3

# Open camera
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run inference
    results = model(frame)[0]
    
    # Process detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        # Check if detected class is a person and above threshold
        if model.names[int(class_id)] == 'person' and score > threshold:
            # Draw bounding box
            cv2.rectangle(frame, 
                          (int(x1), int(y1)), 
                          (int(x2), int(y2)), 
                          (0, 255, 0), 4)
            
            # Add label
            label = f'Person: {score:.2f}'
            cv2.putText(frame, label, 
                        (int(x1), int(y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 3, 
                        cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Person Detection', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()