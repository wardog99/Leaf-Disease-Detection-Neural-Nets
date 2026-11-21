# Potato Leaf Disease Detection - Project Flow

This file summarizes the project workflow implemented in `pld.ipynb` so far.

## 1. Project Setup

- Created a Python notebook for potato leaf disease detection using neural networks.
- Attempted to install `opencv-python`.
- For compatibility, the notebook also attempts to install:
  - `numpy==1.23.5`
  - `opencv-python<4.10`

## 2. Import Required Libraries

The notebook imports:

- `numpy` for array operations.
- `pandas` for dataset and tabular data management.
- `matplotlib.pyplot` and `seaborn` for visualization.
- `os` for file and directory operations.
- `cv2` for image reading and resizing.
- `tensorflow.keras.utils.to_categorical` for label encoding later.

## 3. Define Image Loading

A `load_img_from_folder()` function is used to read the dataset.

For each class folder:

1. The folder name is used as the image label.
2. The notebook finds image files inside the folder.
3. OpenCV reads each image using `cv2.imread()`.
4. Invalid or unreadable images are skipped.
5. Valid images are resized to `128 x 128` pixels.
6. Each image and its label are stored as a `(image, label)` pair.

The notebook is designed to handle three labels:

- `early_blight`
- `late_blight`
- `healthy`

## 4. Load Dataset Splits

The notebook defines local paths for three dataset directories:

- `Training`
- `Testing`
- `Validation`

It calls `load_img_from_folder()` for all three directories and stores the results in:

- `train_data`
- `test_data`
- `val_data`

## 5. Combine Data and Create a DataFrame

The notebook combines the testing, training, and validation samples:

```python
all_data = test_data + train_data + val_data
df = pd.DataFrame(all_data, columns=["image", "label"])
```

The resulting DataFrame has two columns:

- `image`: the image array.
- `label`: the disease or healthy class name.

The notebook reports a total of 4072 samples.

## 6. Resize Images for the Neural Network

Each image is initially loaded at `128 x 128 x 3`.

The notebook checks the dimensions and size of the first image, then resizes every DataFrame image to `224 x 224` pixels:

- Previous shape: `(128, 128, 3)`
- New shape: `(224, 224, 3)`

## 7. Split the Data

Using `train_test_split` from scikit-learn:

- 80% of the combined data is assigned to training.
- 20% is assigned to testing.
- `random_state=42` is used to make the split reproducible.

The resulting split is:

- Training: 3257 samples.
- Testing: 815 samples.

The image and label columns are separated into:

- `train_images`
- `train_labels`
- `test_images`
- `test_labels`

The image lists are converted into NumPy arrays with `np.stack()`.

## 8. Normalize and Format Images

The training and testing image arrays are reshaped to:

- `(number_of_samples, 224, 224, 3)`

Pixel values are scaled from the original range `0-255` to `0-1` by dividing by `255`.

The arrays are converted to `float32` for neural-network use.

## 9. Encode Class Labels

Text labels are converted to numeric values using `LabelEncoder`.

The encoder is fitted using the training labels and then used to transform both training and testing labels.

The numeric labels are then converted to one-hot categorical vectors using `to_categorical()`.

This is the required format for multi-class neural-network classification.

## 10. Preview a Training Sample

A `display_img()` function is defined to:

1. Print the one-hot label for a selected training sample.
2. Display the corresponding normalized image.
3. Hide the plotting axes.

The notebook currently displays sample number 10 as a test of the data preparation workflow.

## Current Status

The notebook has completed data discovery and preprocessing:

- Library setup
- Image loading
- Image resizing
- DataFrame creation
- Train/test splitting
- Pixel normalization
- Label encoding
- One-hot encoding
- Sample visualization

The notebook has not yet built, compiled, trained, or evaluated a neural-network model.