# 🤟 Sign Language Detection using CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

## 📌 Overview

This project focuses on recognizing **American Sign Language (ASL)** gestures using **Convolutional Neural Networks (CNN)** and **image processing techniques**. The system detects hand gestures in real-time via webcam and classifies them into corresponding **textual outputs** using a trained deep learning model.

---

## 🎯 Objectives

- Bridge the communication gap between deaf/mute individuals and others.
- Translate static hand gestures into readable text.
- Build a robust model that can run in real-time.

---

## 📷 Demo

![Sign Language Demo](https://github.com/yourusername/sign-language-cnn/assets/demo.gif)

---

## 🛠️ Tech Stack

| Technology | Description |
|------------|-------------|
| Python     | Programming Language |
| OpenCV     | For image and video processing |
| TensorFlow/Keras | For building CNN model |
| NumPy      | Numerical operations |
| Matplotlib | Visualization |

---

## 📁 Directory Structure

```
sign-language-cnn/
│
├── dataset/                 # Training image dataset
├── model/                   # Saved CNN models
├── utils/                   # Helper functions for preprocessing
├── sign_detect.py           # Real-time detection script
├── train_model.py           # CNN training script
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

---

## 📸 Image Processing Pipeline

1. Capture frame from webcam
2. Convert to grayscale
3. Apply Gaussian Blur
4. Apply thresholding or segmentation
5. Detect contours or bounding boxes
6. Resize and normalize image
7. Feed to CNN model for classification

---

## 🧠 Model Architecture (CNN)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 letters in the alphabet
])
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sign-language-cnn.git
cd sign-language-cnn

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### ▶️ Train the Model
```bash
python train_model.py
```

### ▶️ Run Detection
```bash
python sign_detect.py
```

> Ensure your webcam is connected and the model is trained or downloaded.

---

## 🧪 Sample Gestures

| Gesture         | Output         |
|----------------|----------------|
| ✋ (Open Palm)  | "Stop"         |
| 🤘 (Rock Sign) | "Rock"         |
| 👋 (Wave)       | "Hi"           |
| 👆 (Point)      | "Please help"  |
| ✌️ (V)          | "Good"         |

---

## 🔮 Future Enhancements

- Convert sign outputs into full sentences
- Add **speech synthesis** (Text-to-Speech)
- Include **facial expressions** and **body posture** in detection
- Expand gesture vocabulary beyond ASL

---

## 🧑‍💻 Contributors

- K. Nikhil – [@nikhil](https://github.com/nikhil)
- N. Sanjay – [@sanjaygithub](https://github.com/sanjaygithub)
- K. Naganishith
- G. Vishnu Vardhan
- Ch. Abhiram

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- SR University
- TensorFlow and Keras Documentation
- OpenCV Community
- [adeshpande3.github.io](https://adeshpande3.github.io/)
