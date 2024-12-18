from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Use the model
results = model.train(data="./train.yaml", epochs=10)  # train the model
