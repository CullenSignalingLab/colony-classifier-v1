import csv
import time
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras

import yaml

# --- Load config from ccc.yml ---
def load_config(yaml_path="ccc.yml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
img_width = config["img_width"]
img_height = config["img_height"]
classify_data_dir = pathlib.Path(config["classify_data_dir"])
best_model_path = config["best_model_path"]
class_names = config["class_names"]

# Load the trained model
model = keras.models.load_model(best_model_path)

def load_and_prep_image(img_path, img_height, img_width):
    img = tf.keras.utils.load_img(
        img_path, color_mode="rgb", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    
    # Do NOT rescale here if your model already includes a Rescaling(1./255) layer.
    # This prevents double rescaling which makes all images look exactly the same.
    # Classification will not work in this case.
    # img_array = img_array / 255.0  # <-- REMOVE this line if Rescaling layer is in the model
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

csv_path = pathlib.Path("classification_results.csv")
start_time = time.time()
num_classified = 0

with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "primary_class"] + list(class_names))
    print(f"\nClassifying images in: {classify_data_dir}")
    class_counts = {name: 0 for name in class_names}
    for img_path in classify_data_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
            try:
                img_array = load_and_prep_image(img_path, img_height, img_width)
                predictions = model.predict(img_array, verbose=0)
                probabilities = tf.nn.softmax(predictions[0]).numpy()
                primary_class = class_names[np.argmax(probabilities)]
                class_counts[primary_class] += 1
                writer.writerow([img_path.name, primary_class] + [f"{prob:.6f}" for prob in probabilities])
                num_classified += 1
                class_count_str = ", ".join(f"{name}: {class_counts[name]}" for name in class_names)
                print(f"\rimages classified: {num_classified} ({class_count_str})", end="", flush=True)
            except Exception as e:
                print(f"\nCould not process {img_path.name}: {e}")

print()  # Newline after progress indicator
elapsed = time.time() - start_time
print(f"\nClassification results written to {csv_path.resolve()}")
print(f"Classified {num_classified} images in {elapsed:.2f} seconds.")
