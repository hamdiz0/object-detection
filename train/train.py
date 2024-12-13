from ultralytics import YOLO
import os

# Load a model
model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'best.pt')
model = YOLO(model_path)

# Use the model
test_results = model.predict(
    source="/win/Hamdi/ml/model/Dataset/test/images", # Test images
    save=True,           # Save prediction visualizations
    conf=0.25,           # Confidence threshold
    save_txt=True,       # Save predictions as txt files
    save_conf=True       # Save confidence scores
)