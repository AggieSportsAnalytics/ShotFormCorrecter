from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO('Yolo-Weights/yolov8n.pt')

    # Train the model
    results = model.train(data='config.yaml', epochs=100, imgsz=640)
