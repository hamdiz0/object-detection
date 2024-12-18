from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Use the model
test_results = model.predict(
    source='test/images',    # specify test set path
    save=True,               # Save prediction visualizations
    conf=0.30,               # Confidence threshold
    save_txt=True,           # Save predictions as txt files
    save_conf=True           # Save confidence scores
)
