import csv
import time
import pathlib
import argparse
import sqlite3
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras

import yaml

# --- Load config from ccc.yml ---
def load_config(yaml_path="ccc.yml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

# Parse command line argument for db file
parser = argparse.ArgumentParser(description="Classify images and store results in a sqlite3 db.")
parser.add_argument('--db', type=str, required=True, help='Path to sqlite3 database file')
args = parser.parse_args()
db_path = pathlib.Path(args.db)

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
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

start_time = time.time()
num_classified = 0

# Create database and table if not exists
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
class_prob_cols = ', '.join([f'{name}_prob REAL' for name in class_names])
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        imagekey TEXT,
        {class_prob_cols},
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

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
            # Compute imagekey by stripping non-alphanumeric characters from filename
            imagekey = re.sub(r'[^A-Za-z0-9]', '', img_path.name)
            # Prepare insert statement
            placeholders = ','.join(['?'] * (2 + len(class_names)))
            insert_sql = f'INSERT INTO classifications (filename, imagekey, {", ".join([f"{n}_prob" for n in class_names])}) VALUES ({placeholders})'
            cursor.execute(insert_sql, [img_path.name, imagekey] + [float(prob) for prob in probabilities])
            num_classified += 1
            class_count_str = ", ".join(f"{name}: {class_counts[name]}" for name in class_names)
            print(f"\rimages classified: {num_classified} ({class_count_str})", end="", flush=True)
        except Exception as e:
            print(f"\nCould not process {img_path.name}: {e}")

conn.commit()
conn.close()

print()  # Newline after progress indicator
elapsed = time.time() - start_time
print(f"\nClassification results written to {db_path.resolve()}")
print(f"Classified {num_classified} images in {elapsed:.2f} seconds.")
