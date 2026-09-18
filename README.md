# 🖊️ Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits from **0 to 9** using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.

The project includes an interactive **Streamlit web application** where users can draw a digit and receive a prediction with its confidence score.

---

## ✨ Features

- 🖊️ Interactive drawing canvas
- 🔍 Handwritten digit prediction
- 📊 Prediction confidence and probability graph
- 🖼️ Displays the processed image
- 🔢 Confusion matrix
- ❌ Misclassified image visualization
- 📈 Training and validation graphs

---

## 🧠 Model

The project uses a CNN with:

```text
Input (28 × 28 × 1)
        ↓
Conv2D (32 filters)
        ↓
MaxPooling
        ↓
Conv2D (64 filters)
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dense (10) + Softmax
```

The model is implemented using **TensorFlow/Keras**.

---

## 📂 Dataset

The project uses the **MNIST handwritten digit dataset**.

| Property | Details |
|---|---|
| Training Images | 60,000 |
| Test Images | 10,000 |
| Image Size | 28 × 28 |
| Image Type | Grayscale |
| Classes | 0–9 |

Pixel values are normalized from **0–255 to 0–1** before training.

---

## 📊 Results

The CNN was trained for **10 epochs** and achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **99.03%** |
| Test Loss | **0.0420** |
| Misclassified Images | **97 / 10,000** |

> Results are based on the MNIST test dataset. Performance on user-drawn or real-world handwriting may vary.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Handwritten-Digit-Recognition.git
```

### 2. Navigate to the project

```bash
cd Handwritten-Digit-Recognition
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

On Windows, if `pip` is not recognized:

```bash
py -m pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

On Windows:

```bash
py -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
Handwritten-Digit-Recognition/
│
├── app.py
├── train_model.py
├── mnist_model.h5
├── accuracy_graph.png
├── loss_graph.png
├── confusion_matrix.png
├── misclassified_images.png
├── requirements.txt
├── README.md
└── .gitignore
```

- `app.py` — Streamlit application
- `train_model.py` — CNN training and evaluation
- `mnist_model.h5` — Trained CNN model
- `requirements.txt` — Required Python libraries

---

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- CNN
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- Seaborn
- Streamlit

---

## 👩‍💻 Author

**Vismaya S**

Computer Science Engineering Graduate