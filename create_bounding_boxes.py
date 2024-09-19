import os
import cv2

# Set the paths

# Your image folder path
image_folder = 'pc_parts'  

# Folder to save annotation files
output_folder = 'annotations'  

# Creates the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the image directory
files = os.listdir(image_folder)
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

# Dictionary to map category names to class indexes
categories = {category: idx for idx, category in enumerate(['cables', 'case', 'cpu', 'gpu', 'hdd', 'headset', 'keyboard', 'microphone', 'monitor', 'motherboard', 'mouse', 'ram', 'speakers', 'webcam'])}

for image_file in image_files:
    # Read the image using OpenCV
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Define the bounding box around the entire image (object fills most of the image)
    center_x, center_y = 0.5, 0.5
    box_width, box_height = 1.0, 1.0

    # Determine the class index from the filename

    # Extract category name from the filename
    class_name = image_file.split('_')[0]  
    class_index = categories[class_name]

    # Create annotation string in YOLO format
    annotation = f"{class_index} {center_x} {center_y} {box_width} {box_height}\n"

    # Save the annotation file
    annotation_file = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.txt')
    with open(annotation_file, 'w') as file:
        file.write(annotation)

print("Annotation files generated successfully!")
