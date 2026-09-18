import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas


# Load the trained CNN model
model = tf.keras.models.load_model("mnist_model.h5")


# ==============================
# TITLE
# ==============================

st.title("🖊️ Handwritten Digit Recognition")
st.write("Draw a digit from 0 to 9 and let the CNN predict it.")


# ==============================
# DRAWING CANVAS
# ==============================

st.subheader("Draw a digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)


# ==============================
# CLEAR DRAWING
# ==============================

if st.button("🧹 Clear Drawing"):
    st.rerun()


# ==============================
# PREDICTION
# ==============================

if st.button("🔍 Predict"):

    if canvas_result.image_data is not None:

        # Get the drawn image
        image = canvas_result.image_data.astype("uint8")

        # Convert to grayscale
        image = Image.fromarray(image).convert("L")

        # Resize to MNIST format
        image = image.resize((28, 28))

        # Convert to NumPy array
        img_array = np.array(image)

        # Normalize pixel values
        img_array = img_array / 255.0

        # Reshape for CNN
        img_array = img_array.reshape(1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(img_array, verbose=0)

        # Find predicted digit
        predicted_label = np.argmax(prediction)

        # Get confidence
        confidence = np.max(prediction) * 100

        # Display prediction
        st.success(f"Predicted Digit: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}%")


        # ==============================
        # PROBABILITY GRAPH
        # ==============================

        st.subheader("Prediction Probabilities")

        probabilities = prediction[0] * 100

        fig, ax = plt.subplots()

        ax.bar(range(10), probabilities)

        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability (%)")
        ax.set_title("CNN Prediction Probabilities")
        ax.set_ylim(0, 100)

        st.pyplot(fig)


        # ==============================
        # IMAGE SEEN BY CNN
        # ==============================

        st.subheader("Image Seen by the CNN")

        fig2, ax2 = plt.subplots()

        ax2.imshow(
            img_array.reshape(28, 28),
            cmap="gray"
        )

        ax2.axis("off")

        st.pyplot(fig2)

    else:
        st.warning("Please draw a digit first.")


# ==============================
# MODEL PERFORMANCE
# ==============================

st.divider()

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Test Accuracy", "99.03%")

with col2:
    st.metric("Test Loss", "0.0420")

with col3:
    st.metric("Misclassified", "97")

with col4:
    st.metric("Training Epochs", "10")


st.write("**Model:** Convolutional Neural Network (CNN)")
st.write("**Dataset:** MNIST")
st.write("**Input Size:** 28 × 28 grayscale image")
st.write("**Classes:** 10 (digits 0–9)")


# ==============================
# TRAINING PERFORMANCE
# ==============================

st.subheader("Training Performance")

st.image(
    "accuracy_graph.png",
    caption="Training and Validation Accuracy"
)

st.image(
    "loss_graph.png",
    caption="Training and Validation Loss"
)


# ==============================
# CONFUSION MATRIX
# ==============================

st.subheader("Confusion Matrix")

st.image(
    "confusion_matrix.png",
    caption="CNN Confusion Matrix"
)


# ==============================
# MISCLASSIFIED EXAMPLES
# ==============================

st.subheader("Misclassified Examples")

st.image(
    "misclassified_images.png",
    caption="Examples of Incorrect Predictions"
)