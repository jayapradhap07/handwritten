
# ============================================================
# PROJECT: HANDWRITTEN DIGIT CLASSIFICATION USING CNN
# Dataset: MNIST
# Classes: Digits 0-9
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files

# 1. Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load the MNIST dataset
# Dataset contains 60,000 training and 10,000 test images.
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# 3. Normalize pixel values and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Input image shape:", x_train.shape[1:])

# 4. Display sample handwritten digits
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.suptitle("Sample MNIST Handwritten Digits")
plt.tight_layout()
plt.show()

# 5. Build CNN model with three convolutional layers
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Convolutional layer 1
    layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convolutional layer 2
    layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convolutional layer 3
    layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# 6. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
model.summary()

# 7. Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Evaluate on the separate test dataset
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 9. Plot training and validation accuracy/loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 10. Predict all test images
y_prob = model.predict(x_test, batch_size=256, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# 11. Classification report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(10)),
    zero_division=0
))

# 12. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

plt.figure(figsize=(9, 7))
plt.imshow(cm, cmap="Blues")
plt.title("MNIST Confusion Matrix")
plt.colorbar()

plt.xticks(range(10))
plt.yticks(range(10))
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")

threshold = cm.max() / 2
for i in range(10):
    for j in range(10):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=8
        )

plt.tight_layout()
plt.show()

# 13. Display predictions for sample test images
plt.figure(figsize=(12, 6))

for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")

    color = "green" if y_pred[i] == y_test[i] else "red"
    plt.title(
        f"Pred: {y_pred[i]} | True: {y_test[i]}",
        color=color
    )
    plt.axis("off")

plt.suptitle("Sample Predictions (Green = Correct, Red = Incorrect)")
plt.tight_layout()
plt.show()

# 14. Save the trained model
model_path = "/content/handwritten_digit_cnn.keras"
model.save(model_path)
print("Model saved to:", model_path)

# 15. Upload your own handwritten digit image
# Use a clear, dark digit on a light background.
print("\nUpload a handwritten digit image (PNG/JPG).")
uploaded = files.upload()

for filename in uploaded.keys():
    try:
        # Load image in grayscale and resize to MNIST dimensions
        img = tf.keras.utils.load_img(
            filename,
            color_mode="grayscale",
            target_size=(28, 28)
        )

        img_array = tf.keras.utils.img_to_array(img)

        # MNIST normally has a black background and white digit.
        # Automatically invert images that have a bright background.
        if np.mean(img_array) > 127:
            img_array = 255.0 - img_array

        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        probabilities = model.predict(img_array, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        plt.figure(figsize=(4, 4))
        plt.imshow(img_array[0].squeeze(), cmap="gray")
        plt.title(
            f"Predicted Digit: {predicted_digit}\n"
            f"Confidence: {confidence:.2f}%"
        )
        plt.axis("off")
        plt.show()

        print("Predicted digit:", predicted_digit)
        print(f"Confidence: {confidence:.2f}%")

    except Exception as error:
        print("Could not process image:", error)

# 16. Download trained model
print("\nDownloading trained model...")
files.download(model_path)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
