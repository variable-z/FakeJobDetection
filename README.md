# Fake Job Postings Detection

This repository contains the full codebase and resources for a machine learning and deep learning pipeline designed to detect fake job postings. The system leverages traditional models like Logistic Regression and XGBoost, along with advanced deep learning models such as LSTM and DistilBERT, to accurately classify job listings as real or fraudulent.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Pre-trained Embeddings](#pre-trained-embeddings)
- [Running the Project](#running-the-project)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Team Contributions](#team-contributions)
- [Reproducibility](#reproducibility)

## Project Overview
Online job portals face an increasing problem of fraudulent postings. This project develops an end-to-end pipeline that preprocesses job data, engineers features, and applies a combination of classical and deep learning models to detect fraudulent postings with high reliability.

The full project report is available [here](reports/Final_Report.pdf).

## Folder Structure
```
DFJP_FakeJobDetection/
│
├── Data/                  # Folder for datasets and external files
│
├── preprocessing.ipynb    # Data preprocessing steps
├── modeling.ipynb          # Model training and evaluation
├── requirements.txt        # List of required Python packages
├── README.md               # Project instructions (this file)
└── ML_REPORT.pdf           # Final report detailing methodology and results
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/variable-z/FakeJobDetection.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Pre-trained Embeddings

This project uses **GloVe 100-dimensional word embeddings** to initialize the LSTM model.  
Due to file size limitations, the GloVe embeddings are not included in this repository.

Please manually download the following file:

- [glove.6B.100d.txt (331 MB)](https://nlp.stanford.edu/data/glove.6B.zip)

After downloading:
- Extract the file.
- Place `glove.6B.100d.txt` inside the following path:  
  ```
  Data/glove.6B/glove.6B.100d.txt
  ```

**Note**: Without this file, the LSTM model cannot be trained properly.

## Running the Project

1. Preprocess the data:
   - Open and run `preprocessing.ipynb`.

2. Train and evaluate models:
   - Open and run `modeling.ipynb`.

3. All trained model outputs, confusion matrices, and performance graphs are generated inside the notebook.

## Models Implemented
- Logistic Regression (Classical)
- XGBoost (Classical)
- LSTM (Deep Learning with GloVe Embeddings)
- DistilBERT (Transformer-based fine-tuning)

## Results

| Model             | Accuracy | Precision | Recall | F1 Score | AUC   |
|-------------------|----------|-----------|--------|----------|-------|
| Logistic Regression | 99.22%  | 93.37%    | 89.60% | 91.45%   | 98.33% |
| XGBoost           | 99.69%   | 96.55%    | 97.11% | 96.83%   | 99.91% |
| LSTM              | 95.39%   | 53.85%    | 32.37% | 40.43%   | 85.51% |
| DistilBERT        | 94.46%   | 53.85%    | 32.37% | 40.43%   | 97.78% |

Further details and analysis can be found in the `ML_REPORT.pdf`.

## Team Contributions
- **Sai Pavan P**: Classical machine learning models (Logistic Regression, XGBoost), evaluation metrics, and visualizations.
- **Madhu Sree K**: Deep learning models (LSTM, DistilBERT), threshold tuning, and advanced model training.
- **Venkata Sai Chandra V**: Data preprocessing, feature engineering, and exploratory data analysis (EDA).

## Reproducibility

The project includes all code, cleaned datasets, trained models, and a detailed setup guide to allow easy reproduction of results.  
Please ensure to manually download the GloVe embeddings as mentioned above.

