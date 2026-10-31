# 🛡️ AI Credit Card Fraud Detection System
### End-to-End Deep Learning (ANN) & Interactive Fintech Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-blueviolet.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 1. Project Introduction
The **AI Credit Card Fraud Detection System** is an end-to-end, production-grade Deep Learning project designed for academic defense, college portfolios, and real-world fintech deployment.

Built on **TensorFlow/Keras**, this system leverages an **Artificial Neural Network (ANN)** combined with **SMOTE (Synthetic Minority Over-sampling Technique)** and **leakage-free preprocessing** to isolate fraudulent transactions occurring within an ultra-skewed dataset where frauds constitute less than **0.172%** of all events.

The system features an interactive, dark navy fintech dashboard created in **Streamlit** with real-time inference, risk-tier scoring, model explainability, and multi-dimensional Plotly analytics.

---

## 💥 2. Problem Statement
Credit card fraud costs financial institutions and cardholders tens of billions of dollars annually. With rapid digitization, transaction volumes have scaled exponentially while fraud patterns have grown increasingly elusive.

Major engineering hurdles in credit card fraud detection include:
1. **Extreme Class Imbalance:** Out of 284,807 transactions in benchmark datasets, only 492 are fraudulent (~0.172%). A naive model predicting all transactions as genuine achieves 99.8% accuracy while failing 100% of frauds.
2. **Data Leakage Risks:** Preprocessing scalers (StandardScaler/RobustScaler) or synthetic oversamplers (SMOTE) applied prior to splitting leak test distribution statistics into the training manifold, resulting in artificially inflated metrics that collapse in production.
3. **High Cost of False Negatives:** Missing a fraudulent event (False Negative) causes immediate financial loss and regulatory liability, making **Recall** and **PR-AUC (Precision-Recall Area Under Curve)** the primary optimization metrics rather than simple Accuracy.

---

## 🎯 3. Project Objectives
- **Zero-Leakage Pipeline:** Implement strict separation between training and test sets; fit scalers and apply SMOTE exclusively on training data.
- **Deep Neural Architecture:** Build and tune a multi-layer Artificial Neural Network with Batch Normalization, Dropout, and Sigmoid probability calibration.
- **Academic & Production Evaluation:** Measure Precision, Recall, F1-Score, ROC-AUC, and PR-AUC with interactive Confusion Matrices and threshold sliders.
- **Live Inference Engine:** Provide real-time manual single-transaction evaluation and batch CSV scoring with plain-language neural decision explainability.
- **Premium Fintech UI/UX:** Deliver a modern dark navy dashboard with cyan/purple neon accents and responsive Plotly visual analytics.

---

## 🛠️ 4. Technologies Used
| Component | Technologies |
| :--- | :--- |
| **Deep Learning** | TensorFlow 2.x, Keras (Sequential API) |
| **Resampling & ML** | Imbalanced-Learn (SMOTE), Scikit-Learn |
| **Data Manipulation** | Pandas, NumPy, SciPy |
| **Serialization** | Joblib, Native Keras (`.keras` / `.h5`) |
| **User Interface** | Streamlit, Custom Modern Dark CSS |
| **Visualization** | Plotly Graph Objects & Express, Matplotlib, Seaborn |

---

## 📂 5. Dataset Description
The system is built for the **Kaggle Credit Card Fraud Detection Dataset**:
- **Dataset Link:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Source:** Machine Learning Group (MLG) - Université Libre de Bruxelles (ULB)
- **Total Records:** 284,807 transactions recorded across 2 days in September 2013 by European cardholders.
- **Class Breakdown:**
  - `Class 0` (Genuine): 284,315 (99.828%)
  - `Class 1` (Fraudulent): 492 (0.172%)
- **Features (30 total inputs):**
  - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
  - `V1 - V28`: 28 principal components obtained through PCA (confidential features transformed for cardholder privacy).
  - `Amount`: Transaction monetary amount in Euros/USD.
  - `Class`: Response variable (1 for Fraud, 0 for Genuine).

> [!NOTE]
> Per rigorous engineering principles, **no synthetic or fake training data is generated**. If `creditcard.csv` is missing from `data/`, clear download instructions are provided.

---

## 🧠 6. ANN Architecture Specification
The Artificial Neural Network is structured as follows:

```
[Input Layer: 30 Features]
         │
         ▼
[Dense Layer: 128 Neurons, ReLU Activation]
         │
         ▼
[Batch Normalization Layer]
         │
         ▼
[Dropout Layer: Rate = 0.30]
         │
         ▼
[Dense Layer: 64 Neurons, ReLU Activation]
         │
         ▼
[Batch Normalization Layer]
         │
         ▼
[Dropout Layer: Rate = 0.20]
         │
         ▼
[Dense Layer: 32 Neurons, ReLU Activation]
         │
         ▼
[Output Layer: 1 Neuron, Sigmoid Activation (Fraud Probability)]
```

### Compilation & Callbacks
- **Optimizer:** `Adam(learning_rate=0.001)`
- **Loss:** `BinaryCrossentropy()`
- **Monitored Metrics:** Precision, Recall, AUC (ROC), PR-AUC, Accuracy
- **EarlyStopping:** `patience=6, restore_best_weights=True`
- **ModelCheckpoint:** Saves optimal weights to `models/fraud_model.keras`
- **ReduceLROnPlateau:** Halves learning rate upon plateauing validation loss

---

## ⚙️ 7. Preprocessing & Leakage Prevention
1. **Stratified Split:** 80% train / 20% test preserving the 0.172% fraud incidence across splits.
2. **Robust Scaling:** `RobustScaler` is fitted **only on the training split** for `Amount` and `Time` (which contain extreme outliers). Test data is transformed using the fitted training parameters.
3. **SMOTE Oversampling:** SMOTE generates synthetic minority instances by interpolating between nearest fraud neighbors. This is executed **only on the training partition**, guaranteeing that the test split reflects real-world untouched transaction proportions.

---

## 📊 8. Model Evaluation
The model is evaluated on the untouched 20% test partition:
- **Accuracy:** ~99.9%
- **Precision:** Measures the accuracy of positive fraud flags.
- **Recall (Sensitivity):** Critical metric capturing the percentage of real frauds caught.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **ROC-AUC & PR-AUC:** Area under Receiver Operating Characteristic and Precision-Recall curves.

All test metrics, confusion matrix numbers, and training curves are saved directly to `models/metrics.json` and `models/history.json` and rendered interactively in the Streamlit UI.

---

## 📁 9. Project Directory Structure
```
Credit-Card-Fraud-Detection/
│
├── app.py                      # Premium Streamlit Dark Fintech Dashboard
├── train.py                    # End-to-end model training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation
├── .gitignore                  # Git ignore rules
│
├── data/                       # Dataset directory
│   ├── .gitkeep
│   └── creditcard.csv          # Kaggle dataset (downloaded)
│
├── models/                     # Saved artifacts
│   ├── .gitkeep
│   ├── fraud_model.keras       # Trained TensorFlow ANN
│   ├── scaler.joblib           # Fitted preprocessor scalers
│   ├── metrics.json            # Authentic evaluation scores
│   └── history.json            # Training epoch logs
│
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── preprocessing.py        # Scaler, SMOTE, and dataset checks
│   ├── model.py                # ANN topology and training loops
│   ├── evaluation.py           # Metrics calculation & Plotly charts
│   └── prediction.py           # Inference engine & explainability
│
├── notebooks/                  # Interactive experimentation
│   └── analysis.ipynb          # Jupyter research notebook
│
└── assets/                     # UI styling
    └── style.css               # Dark navy fintech custom CSS
```

---

## 🚀 10. Installation & Setup Instructions

### Step 1: Clone or Navigate to Project
```bash
cd Credit-Card-Fraud-Detection
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Visit the Kaggle dataset page: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download and unzip `creditcard.csv`.
3. Move `creditcard.csv` into the `data/` directory:
   ```
   Credit-Card-Fraud-Detection/data/creditcard.csv
   ```
   *(Alternatively, use the Kaggle CLI: `kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip`)*

---

## 💻 11. How to Run the Project

### Train the Deep Learning Model
Run the automated end-to-end training pipeline:
```bash
python train.py
```
*Optional Flags:*
```bash
python train.py --epochs 25 --batch_size 1024 --scaler robust --patience 6
```

### Launch the Streamlit Fintech Dashboard
Launch the web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ 12. Dashboard Pages Overview
1. **Dashboard Overview:** Real-time KPI counters (Total, Genuine, Fraud, Fraud %), class distribution donut chart, log-scaled amount histograms, and 24h temporal trends.
2. **Transaction Analysis:** Deep exploratory analysis, Pearson correlation rankings with fraud, feature distribution boxplots, and multi-dimensional scatter manifolds.
3. **Fraud Detection (Live Inference):**
   - Quick benchmark loading (Sample Genuine, Sample Fraud, Suspicious High-Risk).
   - Full manual input for all 30 features.
   - Interactive gauge probability meter with decision thresholds.
   - Natural language explainability highlighting high-deflection PCA components.
4. **Deep Learning Model:** Complete neural network architecture diagram, authentic confusion matrix, ROC curve, Precision-Recall curve, and training history curves.
5. **Dataset Explorer:** Dynamic search and filter by amount range and transaction type, with CSV export.
6. **About Project:** Mathematical foundations, project background, and academic submission notes.

---

## 📜 13. License & Credits
Developed as an open-source educational and research project for Credit Card Fraud Detection using Deep Artificial Neural Networks. Released under the **MIT License**.
