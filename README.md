# RWF-2000 Video Violence Detection System (v3.0 - X3D Temporal Analysis)

## Project Summary
This project delivers a robust, real-time solution for automatic violence detection in video streams. The system utilizes **Transfer Learning** on the **X3D-M (eXpanded 3D Convolutional Network)** architecture to analyze motion and temporal cues across 16-frame clips, achieving high accuracy on the **RWF-2000 dataset**.

The final V3.0 model is deployed as a user-friendly, real-time CCTV Monitoring System built with **Python** and **Tkinter**.

---

## 🚀 Key Features

- **Temporal Awareness:** Utilizes a 3D CNN (X3D-M) to analyze motion and sequences, resolving the critical flaw of previous 2D-based models.  
- **High Accuracy:** Achieved **96.50% Test Accuracy** and a high **Violence Recall (97.00%)** on the RWF-2000 test set.  
- **Transfer Learning:** Initialized with Kinetics pre-trained weights for superior feature extraction.  
- **Real-Time Deployment:** Deployed in a standalone Python/Tkinter application for live webcam monitoring and prediction.  

---

## 🛠️ Technology Stack

| Component     | Technology        | Role                                      |
|---------------|-----------------|-------------------------------------------|
| Model         | X3D-M (3D CNN)   | Core video feature extractor              |
| Framework     | PyTorch, Torchvision | Training, Inference, and Model Utilities |
| Deployment    | Python, Tkinter, OpenCV | Real-time webcam capture, GUI, and prediction display |
| Data Handling | NumPy, cv2       | Offline video pre-processing and frame buffering |

---

## 📈 Final Performance (V3.0)

The model was evaluated on a dedicated test set (400 clips) after **100 epochs of fine-tuning** the X3D-M backbone.

| Metric        | Training Loss | Test Loss | Test Accuracy | F1-Score |
|---------------|---------------|-----------|---------------|----------|
| Result        | 0.5283        | 0.1163    | 96.50%        | 96.50%   |

### Classification Report

| Class       | Precision | Recall (Sensitivity) | F1-Score |
|------------|-----------|--------------------|----------|
| NonViolence | 0.9697    | 0.9600             | 0.9648   |
| Violence    | 0.9604    | 0.9700             | 0.9652   |

**Confusion Matrix:**  
The high Recall for the 'Violence' class (**97.00%**) indicates a highly effective system for minimizing missed incidents (False Negatives).

---

## 🛣️ Project Evolution (V1.0 → V3.0)

The final system is the result of resolving a series of critical development roadblocks over a 30-day work period:

- **V1.0 (R3D-18 / PyTorch):** Failed due to a **Critical Data Flaw** — inability to reliably load raw video files, resulting in training on random placeholder tensors.  
- **V2.0 (MobileNetV2 / Keras):** Achieved high accuracy (~96%) but suffered from the **Critical Temporal Flaw**, treating videos as static image collections and failing to analyze motion.  
- **V3.0 (X3D-M / PyTorch):** **Resolution.** Implemented a stable `.npy` pre-processing pipeline and shifted to the 3D X3D-M architecture, successfully capturing temporal dependencies and delivering high, validated performance.

---

## 💻 Installation and Setup

### Prerequisites
You need **Python 3.8+** and the following libraries. It is highly recommended to use a virtual environment.

```bash
# Required dependencies (full list in requirements.txt)
pip install torch torchvision torchaudio opencv-python numpy scikit-learn
