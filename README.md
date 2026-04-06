# ✍️ Offline Signature Verification System

A robust biometric system for 1:N offline handwritten signature identification. This project uses Computer Vision and structural analysis to distinguish between genuine signatures and forgeries.

## 🚀 Key Features
* **Advanced Preprocessing:** Alpha-channel blending, Otsu's thresholding, and morphological operations (Closing & Opening) to clean scanned documents.
* **Smart Segmentation:** Automatic Region of Interest (ROI) extraction using Bounding Box contouring.
* **Combined Biometric Scoring:** Uses a weighted combination of **SSIM** (Structural Similarity Index, 85%) and normalized **MSE** (Mean Squared Error, 15%) to evaluate structural geometry rather than just pixel overlap.
* **Audit Trail:** Built-in SQLite database for detailed logging of processing time, scores, and system decisions.

## 📊 System Diagnostics & Results

### 1. Score Distribution (Genuine vs. Impostor)
The system effectively separates genuine signatures from forgeries using a strict combined score threshold.
*(Upload your `score_distribution.png` to the repo and replace this text with the image link)*

### 2. Confusion Matrix
Demonstrates the 1:N identification accuracy across multiple users.
*(Upload your `confusion_matrix.jpg` to the repo and replace this text with the image link)*

### 3. Structural Difference Mapping
Visualizes the exact areas of geometric mismatch between an enrolled profile and a tested signature.
*(Upload your `example_WORST_MISMATCH.jpg` to the repo and replace this text with the image link)*

## 🛠️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/presko27/Signature-Verification.git](https://github.com/presko27/Signature-Verification.git)
