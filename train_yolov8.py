from ultralytics import YOLO
from clearml import Task

# Initialize ClearML Task
task = Task.init(project_name="PC Parts Classification", task_name="YOLOv8 Training")


if __name__ == "__main__":
    # Load YOLOv8 (nano) model
    model = YOLO('yolov8n.yaml') 

    # Define dataset and hyperparameters
    data_config = 'yolo_dataset/data.yaml'  
    epochs = 5
    img_size = 256

    # Train the model
    model.train(data=data_config, epochs=epochs, imgsz=img_size)

    # Save the model weights
    model.save('yolov8_pc_parts.pt')

    # Finalize ClearML Task
    task.close()

    print("Training completed.")
