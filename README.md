# Tuberculosis Detection System (TB-Net Lite)

An end-to-end deep learning framework for automated Tuberculosis (TB) screening from Chest X-Ray (CXR) radiographs, featuring a modernized TensorFlow 2 / Keras implementation, 5-Fold Cross Validation, and an interactive diagnostic inference CLI with visual risk analysis.

---

## 📌 Overview

Tuberculosis remains one of the world's leading infectious causes of mortality. Early and accessible screening using computer-aided detection (CAD) on digital chest radiographs can significantly relieve healthcare burdens, especially in resource-constrained environments.

This repository implements **TB-Net Lite**, a lightweight convolutional deep neural network tailored for binary chest X-ray classification (**Normal** vs. **Tuberculosis**). It provides:
- A full reproducible training and cross-validation pipeline in Jupyter Notebook ([`Tb-net.ipynb`](Tb-net.ipynb)).
- Production-ready trained weights in Keras 3 format ([`best_tbnet_model.keras`](best_tbnet_model.keras)) and legacy HDF5 ([`best_tbnet_model.h5`](best_tbnet_model.h5)).
- An automated CLI and interactive inference tool with report generation and probability plots ([`model ai/contoh model ai.py`](model%20ai/contoh%20model%20ai.py)).
- Real sample radiograph test images ([`sample_test_xray.jpg`](sample_test_xray.jpg) and [`sample_test_normal.jpg`](sample_test_normal.jpg)).

---

## 🔬 Model Performance & Evaluation

The model was evaluated on a dedicated held-out test split of 750 chest radiographs (375 Normal, 375 Tuberculosis). In medical triage, minimizing False Negatives is critical to avoid missing active infections.

### Test Split Results

| Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Sensitivity / Recall (TB)** | **100.00%** | **0 False Negatives** (375 out of 375 TB cases detected) |
| **ROC - AUC** | **99.44%** | Exceptional discriminative power between classes |
| **F1-Score (TB)** | **82.06%** | Robust harmonic balance of precision and recall |
| **Overall Accuracy** | **78.13%** | High baseline screening accuracy |
| **Specificity (Normal)** | **56.27%** | Conservative triage prioritizing sensitivity |

### Confusion Matrix on Test Set
- **True Positives (TB correctly diagnosed):** 375
- **False Negatives (Missed TB cases):** **0**
- **True Negatives (Normal correctly identified):** 211
- **False Positives (Normal flagged for triage):** 164

*Note: In addition to the single test split, 5-Fold Cross Validation was performed across the dataset (results cataloged in [`cross_validation_results/`](cross_validation_results/)).*

---

## 📂 Project Structure

```text
├── Tb-net.ipynb                     # Main Jupyter Notebook (data prep, training, 5-fold CV, evaluation)
├── best_tbnet_model.keras           # Best trained model weights (Keras 3 modern format)
├── best_tbnet_model.h5              # Best trained model weights (HDF5 format)
├── sample_test_xray.jpg             # Real Tuberculosis sample X-ray for testing
├── sample_test_normal.jpg           # Real Normal healthy sample X-ray for testing
├── ensemble_info.json               # Model ensemble and training metadata
├── cross_validation_results/        # 5-Fold Cross Validation metric summaries & logs
│   ├── kfold_cv_results.csv
│   └── kfold_cv_summary.csv
├── kfold_models/                    # Saved checkpoint weights for each CV fold (Fold 1–5)
├── model ai/                        # Inference application module
│   ├── contoh model ai.py           # CLI & interactive prediction engine
│   └── results/                     # Output directory for diagnostic plots & CSV summaries
├── tbnet_output/                    # Checkpoint exports from notebook training
└── README.md                        # Project documentation (this file)
```

---

## ⚙️ Installation & Requirements

Ensure you have **Python 3.9+** (tested on Python 3.10 – 3.12).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/klayds11/Tuberculosis-Detector.git
   cd Tuberculosis-Detector
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow opencv-python pillow numpy pandas matplotlib seaborn scikit-learn
   ```

---

## 🚀 Quick Start: Running Predictions

You can run automated diagnosis on any chest X-ray image using [`model ai/contoh model ai.py`](model%20ai/contoh%20model%20ai.py).

### 1. Test a Single Image with Visual Diagnosis Plot
```bash
python "model ai/contoh model ai.py" --input sample_test_xray.jpg --visualize
```
**Example CLI Output:**
```text
🔬 Starting Tuberculosis Detection System...
📂 Loading Keras model from: best_tbnet_model.keras
✅ Keras model loaded successfully!

📊 Image Analysis: sample_test_xray.jpg
🔄 Preprocessing image: sample_test_xray.jpg

==================================================
🔴 DETECTION RESULT:
==================================================
📁 File: sample_test_xray.jpg
🏥 Diagnosis: TUBERCULOSIS
📈 Confidence: 99.26%
📊 Normal Probability: 0.0074
📊 TB Probability: 0.9926
⚠️  Risk Level: HIGH

💡 Recommendation:
   Tuberculosis presence suspected. Immediate specialist consultation recommended.
==================================================
📊 Visualization saved to: model ai/results/visualisasi_sample_test_xray.png
✅ CSV summary saved to: model ai/results/hasil_sample_test_xray.csv
```

### 2. Test a Normal / Healthy Chest X-Ray
```bash
python "model ai/contoh model ai.py" --input sample_test_normal.jpg --visualize
```
- **Diagnosis:** 🟢 **NORMAL** (Confidence: `87.06%`, Low Risk)

### 3. Interactive Shell Mode
Run without arguments to enter interactive mode:
```bash
python "model ai/contoh model ai.py"
```

### 4. Batch Prediction across an Entire Directory
```bash
python "model ai/contoh model ai.py" --input path/to/images_folder/ --output model ai/results
```

---

## 📊 Dataset

The model is trained on the publicly benchmarked **Tuberculosis (TB) Chest X-ray Database** hosted on Kaggle:
- **Source:** [Tuberculosis (TB) Chest X-ray Database on Kaggle](https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- **Classes:** Normal (healthy lung radiographs) & Tuberculosis (radiographs exhibiting active tuberculosis lesions).
- **Directory layout:** Images are structured into `Train/`, `Validation/`, and `Test/` partitions.

> *Note: Due to GitHub's file size policies, raw image datasets (>700 MB) are omitted from git tracking. Download the dataset from the Kaggle link above and place it in the `Dataset/` folder if you wish to retrain the notebook.*

---

## 🧠 Model Architecture: TB-Net Lite

The architecture is an efficient convolutional neural network built for high-throughput screening:
- **Input Resolution:** `224 × 224 × 3` with standard `[0, 1]` rescaling.
- **Feature Extraction:** Initial large-receptive-field convolution (`7 × 7`, 32 filters, stride 2) with He-normal initialization, Batch Normalization, and Spatial Dropout.
- **Depthwise Separable Blocks:** Cascaded `SeparableConv2D` layers (64, 128, 256 filters) reducing computational cost while capturing multi-scale anatomical thoracic features.
- **Regularization:** L2 weight regularization (`1e-4`), Spatial Dropout (`0.1`–`0.2`), and Batch Normalization.
- **Classifier Head:** Global Average Pooling (`GlobalAveragePooling2D`) followed by Dense projection and a single Sigmoid activation unit outputting TB likelihood.

---

## 📖 Attribution & Acknowledgements

This project was developed by building upon and adapting concepts from the following prior research:

1. **Original Research Paper:**
   > Alexander Wong, James Ren Hou Lee, Hadi Rahmat-Khah, Ali Sabri, and Amer Alaref.  
   > *"TB-Net: A Tailored, Self-Attention Deep Convolutional Neural Network Design for Detection of Tuberculosis Cases from Chest X-ray Images."*  
   > arXiv preprint [arXiv:2104.03165](https://arxiv.org/abs/2104.03165), 2021.

2. **Original Reference Codebase:**
   > DarwinAI Team: [DarwinAI / TuberculosisNet](https://github.com/darwinai/TuberculosisNet).

3. **Key Modifications in this Project:**
   - **Framework Modernization:** Re-engineered the training pipeline from legacy TensorFlow 1.15 checkpoint mechanics to standard, modular **TensorFlow 2 / Keras 3** (`.keras`).
   - **Architecture Adaptation:** Designed a streamlined **TB-Net Lite** network optimized for rapid training and inference without specialized custom graph operations.
   - **Validation Protocol:** Implemented a rigorous 5-fold cross-validation scheme to assess generalizability across patients.
   - **Inference & Diagnostic Tooling:** Developed an end-to-end Python CLI and visualization tool that generates publication-ready probability plots and structured diagnostic CSV logs.

---

## ⚖️ License & Disclaimer

### Disclaimer
This software is intended **solely for educational, academic, and research purposes**. It is not cleared as a medical device for direct clinical diagnosis. Any automated classification output must be interpreted by a licensed radiologist or healthcare professional.

### License
In accordance with open-source research distribution, this project is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
