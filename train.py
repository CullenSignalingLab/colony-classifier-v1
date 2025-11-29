import os
import pathlib
import csv
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

import yaml

# --- Load config from ccc.yml ---
def load_config(yaml_path="ccc.yml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
img_width = config["img_width"]
img_height = config["img_height"]
batch_size = config["batch_size"]
epochs = config["epochs"]
training_data_dir = pathlib.Path(config["training_data_dir"])
model_path = config["model_path"]
best_model_path = config["best_model_path"]
learning_rate = config["learning_rate"]

print(training_data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(
    training_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
print(train_ds.class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    training_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Comment out data_augmentation if you want it disabled, otherwise leave as is.
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ]
)

base_model = MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base model

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),  # For MobileNetV2: scale to [0, 1]
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(train_ds.class_names))
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# Check class balance
def count_images_per_class(directory):
    counter = Counter()
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            counter[class_dir] = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ])
    return counter

print("Images per class:", count_images_per_class(str(training_data_dir)))

# Compute class weights if imbalance exists
labels = []
for class_idx, class_dir in enumerate(train_ds.class_names):
    class_path = training_data_dir / class_dir
    labels.extend([class_idx] * len(list(class_path.glob("*"))))
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(labels), y=labels)))
print("Class weights:", class_weights)

# Add callbacks for early stopping and model checkpointing
callbacks = [
    keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save the trained model for later use in prediction/classification
model.save(model_path)

# Output the classification of each training file
print("\nClassifying training images and writing results to training_classification.csv")
csv_path = pathlib.Path("training_classification.csv")
class_names = train_ds.class_names

with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filepath", "primary_class"] + list(class_names))
    # Get the list of file paths from the dataset
    file_paths = []
    for file_batch in train_ds.file_paths if hasattr(train_ds, 'file_paths') else []:
        file_paths.append(file_batch)
    file_path_idx = 0
    for batch in train_ds:
        images, labels = batch
        predictions = model.predict(images, verbose=0)
        probabilities = tf.nn.softmax(predictions, axis=1).numpy()
        for i in range(images.shape[0]):
            # Get the full relative path from the dataset if available
            if hasattr(train_ds, 'file_paths'):
                rel_path = pathlib.Path(train_ds.file_paths[file_path_idx]).relative_to(training_data_dir.parent)
                file_path_idx += 1
                filepath_str = str(rel_path)
            else:
                filepath_str = f"train_image_{i}"
            probs = probabilities[i]
            primary_class = class_names[np.argmax(probs)]
            writer.writerow([filepath_str, primary_class] + [f"{prob:.6f}" for prob in probs])

print(f"Training classification results written to {csv_path.resolve()}")

# After training, print confusion matrix for training data
all_labels = []
all_preds = []
for batch in train_ds:
    images, labels = batch
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    all_labels.extend(labels.numpy())
    all_preds.extend(preds)
print("Training classification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
