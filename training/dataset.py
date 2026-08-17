import kagglehub
from pathlib import Path
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
IMAGE_SIZE = (128, 128)

required_path = Path("./data/raw")
raw_path = required_path / "data"
if not raw_path.exists():
    path = kagglehub.dataset_download(
        "mahmoudreda55/satellite-image-classification",
        output_dir=str(required_path),
        force_download=False,
    )
    print("Path to dataset files:", path)

raw_path = str(raw_path)
output_path = "./data/processed"
classes = sorted(os.listdir(raw_path))
print("classes: ", classes)


def load_images(data_dir, target_size=(64, 64)):
    X = []
    y = []
    
    # Extract class labels from subfolder names
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
            
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            
            try:
                # Load image and convert to RGB
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    # Resize to ensure all feature dimensions match
                    img = img.resize(target_size)
                    
                    # Convert to NumPy array and flatten into a 1D vector
                    img_array = np.array(img).flatten()
                    
                    X.append(img_array)
                    y.append(class_to_idx[cls_name])
            except Exception as e:
                print(f"Skipping corrupt image {img_path}: {e}")

    # Convert native Python lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def save_split_images(X_data, y_data, output_dir, classes, target_size=(64, 64)):
    """
    Reshapes flattened image vectors and saves them into class-named folders.
    """
    # Create the root folder for this split (e.g., 'output/train')
    os.makedirs(output_dir, exist_ok=True)
    
    # Track image counts per class to give files unique names
    class_counters = {i: 0 for i in range(len(classes))}
    
    # Expected 3D shape: (Height, Width, RGB Channels)
    img_shape = (target_size[0], target_size[1], 3)
    
    for features, label_idx in zip(X_data, y_data):
        cls_name = classes[label_idx]
        class_folder = os.path.join(output_dir, cls_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # 1. Reshape the 1D flat vector back into a 3D matrix
        img_array = features.reshape(img_shape)
        
        # 2. Convert pixel data to unsigned 8-bit integers (required by PIL)
        img_array = img_array.astype(np.uint8)
        
        # 3. Reconstruct the image object and save
        img = Image.fromarray(img_array)
        
        img_name = f"img_{class_counters[label_idx]}.jpg"
        img.save(os.path.join(class_folder, img_name))
        
        class_counters[label_idx] += 1



X, y = load_images(raw_path, target_size=IMAGE_SIZE)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=SEED,
    stratify=y  # Maintains class distribution balance
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    random_state=SEED,
    stratify=y_temp
)

# Save the training set
save_split_images(X_train, y_train, os.path.join(output_path, 'train'), classes, target_size=IMAGE_SIZE)
save_split_images(X_val, y_val, os.path.join(output_path, 'validation'), classes, target_size=IMAGE_SIZE)
save_split_images(X_test, y_test, os.path.join(output_path, 'test'), classes, target_size=IMAGE_SIZE)
