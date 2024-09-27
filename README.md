# 🐱 Cats vs. Dogs Image Classification with CNN 🐶

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. Using TensorFlow and Keras, we train a deep learning model to accurately differentiate between images of these two animals. The dataset comes from the [Kaggle Cats and Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). The project is divided into several key sections, explained below.


## 🚀 Project Setup

### 1. **Imports and Setup**

- **TensorFlow Installation:** First, install TensorFlow with GPU support:
  ```bash
  %pip install tensorflow-gpu
  ```
- **Necessary Imports:** The project requires TensorFlow, Keras, Pandas, and several utilities for data preprocessing and augmentation.
- **Configuration:** Paths for the dataset and training parameters (image size, batch size, etc.) are defined.
- **GPU Check:** Ensure GPU acceleration is available:
  ```python
  print(tf.config.list_physical_devices('GPU'))
  ```

---

## 🛠 Preprocessing

### 2. **Dataset Preprocessing**

This section covers downloading, extracting, and preparing the dataset for training. Key operations include:
- **Dataset Download:** The dataset is downloaded from the provided URL.
- **File Extraction:** Extracts the contents from the downloaded zip file.
- **Image Verification:** Ensures only valid image files are used.
- **Dataset Split:** Splits the dataset into training and testing sets (80% training, 20% testing).

Example code for preprocessing:
```python
prc = Prc()

# Downloading the dataset
prc.download_dataset(
  'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip',
  'catsVsdogs.zip'
)

# Extracting the zip file
prc.extract_zip('/content/catsVsdogs.zip', '/content/')

# Verifying images and splitting the dataset
prc.image_verification('/content/PetImages')
prc.split_dataset('/content/PetImages', '/content/dataset/')
```

---

## 📈 Data Augmentation

### 3. **Data Augmentation**

To prevent overfitting and enhance model generalization, we apply various image transformations such as:
- **Zooming**
- **Flipping (horizontal and vertical)**
- **Rotation**
- **Shifting and shearing**

These transformations are performed using the `ImageDataGenerator`:
```python
train_data_generator = ImageDataGenerator(
    rescale=1.0/255.0,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.4,
    fill_mode='nearest',
)

test_data_generator = ImageDataGenerator(rescale=1.0/255.0)
```

---

## 🧠 Model Architecture

### 4. **Model Definition**

We use a sequential CNN model with several convolutional, pooling, and dropout layers for regularization. The architecture includes:
- Convolutional layers with `ReLU` activation
- Pooling layers to reduce dimensionality
- Dropout layers for regularization
- A final dense layer with a sigmoid activation for binary classification.

```python
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## ⏳ Callbacks

### 5. **Model Callbacks**

We utilize various callbacks to optimize training:
- **EarlyStopping:** Stops training when validation loss no longer improves.
- **ReduceLROnPlateau:** Reduces learning rate when validation loss plateaus.
- **ModelCheckpoint:** Saves the best model based on validation performance.

```python
es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
rlrop = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=0.001)
checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', mode='min', save_best_only=True)
```

---

## 🏋️‍♂️ Model Training

### 6. **Training the Model**

The model is trained using the training and test data generators, with a specified number of epochs and the callbacks defined earlier.

```python
history = model.fit(
    train,
    validation_data=test,
    epochs=100,
    callbacks=[es, rlrop, checkpoint]
)
```

---

## 📊 Results & Visualization

### 7. **Plotting Training History**

We plot the training and validation loss and accuracy over epochs to observe how the model learns over time:

```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
```

---

## 💾 Model Saving

### 8. **Saving the Model**

The trained model is saved to a specified directory:

```python
model.save('/content/models/model.keras')
```

---

## 🎉 Conclusion

This project demonstrates how to build, train, and evaluate a CNN for the task of image classification using a dataset of cats and dogs. By applying data augmentation, callbacks, and a well-structured model architecture, we achieve a robust solution to this binary classification problem.

Feel free to explore and modify the notebook to further enhance the model or apply it to new datasets!

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, request new features, or suggest improvements via pull requests.

---

## 📝 License

This project is licensed under the MIT License.