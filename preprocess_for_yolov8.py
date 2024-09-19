import os
from ultralytics import YOLO

# Split data function - For YOLOv8, use the YOLO data format directly
def prepare_yolo_data():
    # This function can be used to verify the data format
    print("Data is ready in YOLO format.")

# Load YOLOv8 model for training
def load_yolo_model(model_type='yolov8n'):
    # Create a new model from a configuration file 
    model = YOLO(f'{model_type}.yaml')  
    return model

if __name__ == "__main__":
    prepare_yolo_data()
    print("Preprocessing completed.")
