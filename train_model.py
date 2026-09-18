import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ==============================
# LOAD MNIST DATASET
# ==============================

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ==============================
# NORMALIZE PIXEL VALUES
# ==============================

x_train, x_test = x_train / 255.0, x_test / 255.0


# ==============================
# ADD CHANNEL DIMENSION
# ==============================

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# ==============================
# BUILD CNN MODEL
# ==============================

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        (2, 2)
    ),

    keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        (2, 2)
    ),

    keras.layers.Flatten(),

    keras.layers.Dense(
        128,
        activation="relu"
    ),

    keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# TRAIN MODEL
# ==============================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)


# ==============================
# EVALUATE MODEL
# ==============================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")


# ==============================
# SAVE TRAINED MODEL
# ==============================

model.save("mnist_model.h5")

print("Model successfully saved as mnist_model.h5")


# ==============================
# TRAINING & VALIDATION ACCURACY
# ==============================

epochs = range(
    1,
    len(history.history["accuracy"]) + 1
)

plt.figure()

plt.plot(
    epochs,
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.legend()

plt.savefig(
    "accuracy_graph.png",
    bbox_inches="tight"
)

plt.close()


# ==============================
# TRAINING & VALIDATION LOSS
# ==============================

plt.figure()

plt.plot(
    epochs,
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()

plt.savefig(
    "loss_graph.png",
    bbox_inches="tight"
)

plt.close()


# ==============================
# CONFUSION MATRIX
# ==============================

# Get predictions for test dataset
predictions = model.predict(
    x_test,
    verbose=0
)

# Convert probabilities to predicted digits
predicted_labels = np.argmax(
    predictions,
    axis=1
)

# Create confusion matrix
cm = confusion_matrix(
    y_test,
    predicted_labels
)


# Plot confusion matrix
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("CNN Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()


# ==============================
# MISCLASSIFIED IMAGES
# ==============================

# Find incorrectly predicted images
incorrect_indices = np.where(
    predicted_labels != y_test
)[0]

print(
    f"Number of misclassified images: "
    f"{len(incorrect_indices)}"
)


# Display first 12 incorrect predictions
plt.figure(figsize=(10, 8))

for i, index in enumerate(
    incorrect_indices[:12]
):

    plt.subplot(3, 4, i + 1)

    plt.imshow(
        x_test[index].reshape(28, 28),
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[index]}\n"
        f"Predicted: {predicted_labels[index]}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "misclassified_images.png",
    bbox_inches="tight"
)

plt.close()


# ==============================
# FINAL RESULTS
# ==============================

print("\n==============================")
print("MODEL TRAINING COMPLETE")
print("==============================")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(
    f"Misclassified Images: "
    f"{len(incorrect_indices)}"
)
print("Training Epochs: 10")

print("\nGenerated files:")
print("- mnist_model.h5")
print("- accuracy_graph.png")
print("- loss_graph.png")
print("- confusion_matrix.png")
print("- misclassified_images.png")