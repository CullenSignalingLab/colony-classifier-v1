# colony-classifier-v1
First version of AI generated colony image classifier system

## Common Keras/TensorFlow Base Models for Image Classification

- **MobileNetV2**: Lightweight, fast, good for small/medium datasets and limited hardware. Often used for transfer learning.
- **EfficientNet (B0–B7)**: State-of-the-art accuracy/efficiency tradeoff. B0 is smallest, B7 is largest. Good for scaling up with more VRAM.
- **ResNet (ResNet50, ResNet101, etc.)**: Classic deep residual networks, robust and widely used.
- **DenseNet (DenseNet121, etc.)**: Uses dense connections for improved feature propagation.
- **InceptionV3**: Good accuracy, moderate size, uses inception modules for multi-scale feature extraction.
- **Xception**: Depthwise separable convolutions, high accuracy, larger than MobileNet.
- **NASNet**: Neural architecture search-based, high accuracy, but large and slow.
- **VGG16/VGG19**: Older, very large, not efficient, but still used for some transfer learning tasks.

**Selection tips:**
- For limited VRAM or fast prototyping: **MobileNetV2** or **EfficientNetB0/B1**
- For best accuracy and more VRAM: **EfficientNetB3/B4/B5**, **ResNet50/101**
- For classic/robust results: **ResNet50**, **InceptionV3**

All are available in `tf.keras.applications`.

# Data Augmentation: Benefits and Trade-offs

**Benefits:**
- **Improved Generalization:** Helps the model learn features that are robust to small changes, reducing overfitting.
- **Synthetic Data Increase:** Effectively increases the size and diversity of the training set without collecting more data.
- **Better Performance on Unseen Data:** Makes the model less sensitive to orientation, position, and minor variations in the input images.

**Trade-offs:**
- **Longer Training Time:** Each epoch takes longer due to on-the-fly image transformations.
- **Potential Underfitting:** If augmentation is too strong, the model may struggle to learn the true features of each class.
- **Parameter Tuning Required:** The strength and type of augmentation must be tuned for your specific dataset; too much or too little can hurt performance.
- **Not Always Needed:** For very clean, standardized scientific images, heavy augmentation may not provide much benefit and can even degrade accuracy.

**Summary:**  
Keep data augmentation if you want better generalization and have limited, non-identical images. Reduce or remove it if your images are already highly standardized or if you observe underfitting.

## What is a "small or medium-sized dataset"?

- **Small dataset:** Fewer than 5,000 labeled images in total (often <1,000 per class).
- **Medium dataset:** Between 5,000 and 50,000 labeled images in total (typically 1,000–10,000 per class).
- **Large dataset:** More than 50,000 labeled images (e.g., ImageNet scale).

For yeast colony classification, most lab datasets are considered "small" or "medium" by these definitions.
