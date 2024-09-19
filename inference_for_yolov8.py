from ultralytics import YOLO
from clearml import Task

# ClearML task setup
try:
    task = Task.init(project_name="PC Parts Classification", task_name="YOLOv8 Inference", reuse_last_task_id=False)
except Exception as e:
    print(f"Error initializing task: {e}")
    Task.current_task().close()
    task = Task.init(project_name="PC Parts Classification", task_name="YOLOv8 Inference", reuse_last_task_id=False)

# Load the trained model
model = YOLO('yolov8_pc_parts.pt')  

# Perform inference
results = model.predict(source='yolo_dataset/images/val', save=True, save_txt=True)  

# Print predictions
for result in results:
    # Bounding boxes
    print(result.boxes)  
    # Class names
    print(result.names)  

# Finalize ClearML Task
task.close()
