
#  Handwritten Digit Recognition Using Multiple Machine Learning Approaches

##  Project Overview

This project presents a comprehensive comparison of multiple machine learning and deep learning techniques for handwritten digit recognition using the **MNIST dataset**. The goal is not only to achieve high classification accuracy but also to analyze the **strengths, limitations, and trade-offs** of different learning paradigms when applied to image data.

The project is part of a **Machine Learning final term assignment** and focuses on understanding how classical models and neural networks handle visual pattern recognition tasks.

---

##  Dataset

* **Dataset:** MNIST Handwritten Digit Dataset
* **Source:** Kaggle
* **Classes:** Digits 0–9
* **Image Size:** 28 × 28 grayscale
* **Training Samples:** 60,000
* **Test Samples:** 10,000

The dataset is loaded directly from raw `.ubyte` files to demonstrate low-level data handling and avoid reliance on black-box loaders.

---

##  Models Implemented

The following models were implemented and evaluated:

###  Classical Machine Learning

* **k-Nearest Neighbors (kNN)**
* **Support Vector Machine (SVM)**

###  Neural Networks

* **Multi-Layer Perceptron (MLP)**
* **Convolutional Neural Network (CNN)**

This progression allows a clear comparison between traditional and deep learning approaches.

---

##  Data Preprocessing

* Raw `.ubyte` files parsed manually
* Pixel values normalized to `[0, 1]`
* Images flattened for classical ML and MLP
* Spatial structure preserved for CNN
* Labels one-hot encoded for neural networks

---

##  Evaluation Metrics

Each model was evaluated using:

* **Accuracy**
* **Training Time**
* **Prediction (Inference) Time**

These metrics provide insight into both **performance** and **computational efficiency**.

---

##  Experimental Results

| Model   | Accuracy (%) | Training Time (s) | Prediction Time (s) |
| ------- | ------------ | ----------------- | ------------------- |
| kNN     | 97.05        | 0.14              | 59.47               |
| SVM     | 94.04        | 361.11            | 68.19               |
| MLP     | 97.76        | 57.69             | 1.03                |
| **CNN** | **99.16**    | **600.36**        | **5.28**            |

---

##  Key Observations

* **kNN** achieves high accuracy but is impractical due to extremely slow inference.
* **SVM** provides strong decision boundaries but suffers from high training cost.
* **MLP** offers a good balance between speed and accuracy.
* **CNN** outperforms all models by learning spatial features, making it the most robust and accurate approach for image classification.

---

##  Project Structure

```
├── Handwritten_Digit_Recognition.ipynb
├── README.md
└── mnistdata/
    ├── train-images.idx3-ubyte
    ├── train-labels.idx1-ubyte
    ├── t10k-images.idx3-ubyte
    └── t10k-labels.idx1-ubyte
```

---

##  How to Run

1. Open the notebook in **Google Colab**
2. Mount Google Drive
3. Place MNIST `.ubyte` files in the specified directory
4. Run cells sequentially to train and evaluate all models

---

##  Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* Google Colab

---

## 🎯 Conclusion

This project demonstrates that deep learning models, particularly CNNs, significantly outperform classical machine learning approaches for handwritten digit recognition. By preserving spatial information and learning hierarchical features, CNNs provide superior accuracy and robustness, making them the preferred choice for image-based classification tasks.


