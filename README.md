**German Traffic Sign Classifier (GTSRB) using CNN**


This project implements a **Convolutional Neural Network (CNN)** to classify images of German traffic signs using the **GTSRB dataset**. It includes full data preprocessing, model training with data augmentation, evaluation, and prediction on real-world images.

---

## 📊Dataset

- **Source:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Classes:** 43 different traffic signs
- **Format:** 32x32 RGB images

- **Download the required files:**
- [train.p](https://drive.google.com/file/d/19vPc9gJ7LPv8Uw-D3gbYa7psLgb-jy0v/view?usp=sharing)
- [test.p](https://drive.google.com/file/d/1lb1JSa2pi-KDvVOAG1pyvMvoIbEOzSt9/view?usp=sharing)
- [valid.p](https://drive.google.com/file/d/13LPCK0VESsKqJxaP0YDZ5yqtAvvB-uQp/view?usp=sharing)
- [signnames.csv](https://drive.google.com/file/d/1Ww3Ns5x6vrlSszfu_X5n8Wnn5bGLbQOJ/view?usp=sharing)
---

## 🧠 Model Architecture

The model is a custom CNN built using Keras:

- 2 x Conv2D (60 filters, 5x5 kernel) + ReLU
- MaxPooling2D
- 2 x Conv2D (30 filters, 3x3 kernel) + ReLU
- MaxPooling2D
- Flatten
- Dense (500) + ReLU
- Dropout (0.5)
- Dense (43, Softmax)

**Compiled with:**

```python
Adam(lr=0.001)
loss='categorical_crossentropy'
metrics=['accuracy']
