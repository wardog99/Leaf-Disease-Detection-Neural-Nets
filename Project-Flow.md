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

- `Early_Blight`
- `Late_Blight`
- `Healthy`

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

## 11. Build the Simple CNN

The latest notebook section adds a TensorFlow/Keras sequential convolutional neural network for the three disease classes.

The model architecture is:

1. `Conv2D(32, 3 x 3, relu)` for the 224 x 224 x 3 input.
2. `Conv2D(32, 3 x 3, relu)`.
3. `GlobalAveragePooling2D()`.
4. `Dense(128, relu)` with L1/L2 regularization (`1e-5`).
5. `Dense(64, relu)`.
6. `Dropout(0.3)`.
7. `BatchNormalization()`.
8. `Dense(3, softmax)` for the three output classes.

The model summary reports:

- Total parameters: 23,075.
- Trainable parameters: 22,947.
- Non-trainable parameters: 128.

## 12. Compile and Train the Model

The model is compiled with:

- Loss: `categorical_crossentropy`.
- Optimizer: Adam with a learning rate of `0.001`.
- Metric: accuracy.

Training uses:

- The prepared training images and one-hot labels.
- A 20% validation split.
- 20 epochs.
- `EarlyStopping` monitoring `val_loss` with a patience of 7 epochs.

The recorded run completed all 20 epochs. The final epoch produced:

- Training loss: `0.5496`.
- Training accuracy: `0.8276`.
- Validation loss: `0.5442`.
- Validation accuracy: `0.7899`.

The best-recorded validation performance appeared at epoch 19:

- Validation loss: `0.5130`.
- Validation accuracy: `0.8313`.

The training curve improved substantially after the initial epochs. Training accuracy increased from `0.4077` at epoch 1 to `0.8276` at epoch 20, while validation accuracy reached `0.8313` at epoch 19.

## 13. Environment and Compatibility Notes

The notebook includes package setup notes and compatibility attempts:

- It attempted to install `opencv-python`.
- It contains a pinned downgrade attempt for `numpy==1.23.5` and `opencv-python<4.10`.
- TensorFlow reported that Adam may run slowly on M1/M2 Macs and used the legacy Keras Adam implementation for the recorded training run.

## 14. Evaluate the Trained Model

The notebook evaluates the trained CNN on the held-out test set:

- Test samples: 815.
- Test loss: `0.5997`.
- Test accuracy: `0.7681` (approximately 76.81%).

Predictions are generated with `model.predict()`, and the probability output for each image is converted to a class using `np.argmax()`. The same operation is used to convert the one-hot test labels back to numeric classes.

## 15. Confusion Matrix and Classification Report

The notebook computes:

- A confusion matrix with scikit-learn.
- Accuracy with `accuracy_score`.
- Precision, recall, and F1-score with `classification_report`.
- A visual confusion-matrix heatmap using Seaborn.

For the recorded `LabelEncoder` ordering, the numeric classes correspond to:

- `0`: `Early_Blight`
- `1`: `Healthy`
- `2`: `Late_Blight`

The test results by numeric class are:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.65 | 0.95 | 0.77 | 303 |
| 1 | 0.94 | 0.38 | 0.54 | 217 |
| 2 | 0.90 | 0.86 | 0.88 | 295 |

Overall test metrics:

- Accuracy: `0.7681`.
- Macro-average precision: `0.83`.
- Macro-average recall: `0.73`.
- Macro-average F1-score: `0.73`.
- Weighted-average precision: `0.82`.
- Weighted-average recall: `0.77`.
- Weighted-average F1-score: `0.75`.

The report shows strong performance for classes 0 and 2, while class 1 has high precision (`0.94`) but a low recall (`0.38`). This indicates that the model correctly identifies many of the class 1 samples it predicts, but misses a substantial portion of the actual class 1 samples.

## Current Status

The notebook has completed data discovery, preprocessing, model construction, training, and test evaluation:

- Library setup
- Image loading
- Image resizing
- DataFrame creation
- Train/test splitting
- Pixel normalization
- Label encoding
- One-hot encoding
- Sample visualization
- Simple CNN architecture
- Model compilation
- Model training
- Validation metric capture
- Held-out test evaluation
- Confusion-matrix visualization
- Classification report

The notebook has not yet added saved-model/export steps.