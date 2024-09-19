import os
import shutil

# Convert annotations to YOLO format
def convert_to_yolo_format(annotations_dir, img_dir, output_dir):
    # Create the output directories
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # List all annotation files in the annotations directory
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    
    for idx, annotation_file in enumerate(annotation_files):
        # Derive the image name from the annotation file name
        image_name = annotation_file.replace('.txt', '.jpg')  
        image_path = os.path.join(img_dir, image_name)
        label_path = os.path.join(annotations_dir, annotation_file)

        # Check if the image exists in the image directory
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_name} not found for annotation {annotation_file}")
            continue

        # Decide whether to put the image in 'train' or 'val' set
        if idx % 5 == 0:
            # Validation set
            shutil.copy(image_path, os.path.join(output_dir, "images", "val", image_name))
            new_label_path = os.path.join(output_dir, "labels", "val", annotation_file)
        else:
            # Training set
            shutil.copy(image_path, os.path.join(output_dir, "images", "train", image_name))
            new_label_path = os.path.join(output_dir, "labels", "train", annotation_file)
        
        # Copy the annotation file to the corresponding directory
        shutil.copy(label_path, new_label_path)

    print("Conversion to YOLO format completed.")

if __name__ == "__main__":
    annotations_dir = 'annotations' 
    img_dir = 'pc_parts'
    output_dir = 'yolo_dataset'
    convert_to_yolo_format(annotations_dir, img_dir, output_dir)
