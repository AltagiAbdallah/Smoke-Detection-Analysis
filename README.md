# Smoke Detection Analysis

## Overview

This repository contains the code for building and evaluating a machine learning model to detect smoke based on sensor data. The project follows a standard machine learning workflow, including data loading, exploration, preprocessing, model training, and evaluation.

## Features

* **Data Loading**: Loads sensor data from a CSV file.
* **Data Exploration**: Provides insights into the dataset's structure, missing values, duplicates, and feature correlations.
* **Target Variable Analysis**: Visualizes the distribution of the 'Fire Alarm' target variable.
* **Data Preprocessing**: Handles data splitting into training and testing sets, and scales numerical features.
* **Model Training**: Implements and trains two popular classification models:
  * Support Vector Machine (SVM)
  * Decision Tree Classifier
* **Model Evaluation**: Evaluates model performance using accuracy, classification reports, and confusion matrices.

## Getting Started

Follow these instructions to set up the project locally and run the code.

### Prerequisites

You need Python 3.x installed. The following libraries are required:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

Install them using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/smoke-detection-ml.git
cd smoke-detection-ml
```

(Replace `your-username/smoke-detection-ml` with your actual GitHub repository path.)

2. **Dataset:**

Place your dataset `smoke_detection_iot.csv` in the appropriate directory. If using Google Colab, the default path is `/content/drive/MyDrive/new/`, otherwise adjust accordingly.

## Usage

The main logic is contained within a Python script or Jupyter Notebook.

To run the full analysis and model training, execute the script:

```bash
python your_model_script_name.py
```

If it's a Jupyter Notebook (`.ipynb`), open it in Jupyter or Google Colab and run all cells.

### Script Sections:

1. **Introduction**: Imports libraries and mounts Google Drive (if in Colab).
2. **Dataset Loading**: Loads `smoke_detection_iot.csv`.
3. **Data Exploration**: Prints head, info, checks for missing values/duplicates, and shows a correlation heatmap.
4. **Dataset Background Analysis**: Shows 'Fire Alarm' class distribution via count plot.
5. **Pre-processing**: Splits dataset (80% train, 20% test) and scales features using `StandardScaler`.
6. **Model Evaluation Technique**: Applies the Hold-out evaluation method.
7. **Model Training and Evaluation**:
   * **SVM**: Trains a Support Vector Classifier, predicts results, and outputs performance metrics.
   * **Decision Tree**: Trains a Decision Tree Classifier, predicts results, and outputs performance metrics.

## Dataset

The project uses the `smoke_detection_iot.csv` dataset.

- **Entries**: 62,630
- **Columns**: 16
- **Target**: `Fire Alarm` (0 = No Fire, 1 = Fire)
- **Features**: Includes `Temperature[C]`, `Humidity[%]`, `TVOC[ppb]`, `eCO2[ppm]`, `Raw H2`, `Raw Ethanol`, `Pressure[hPa]`, `PM2.5`, `NC2.5`, `CNT`, `UTC`, `PM1.0`, `NC0.5`, `NC1.0`.

## Results

The models are evaluated based on:

* **Accuracy Score**
* **Classification Report** (Precision, Recall, F1-Score)
* **Confusion Matrix**

All performance metrics and plots are displayed as output during execution.

## Visualizations

To save and include visualizations in your README, use `plt.savefig('filename.png')` before `plt.show()` in your code.

Ensure you create an `images` folder in the repo and upload the images there.

### 1. Correlation Matrix

```markdown
![Correlation Matrix](images/correlation_matrix.png)
```

### 2. 'Fire Alarm' Class Distribution

```markdown
![Fire Alarm Class Distribution](images/class_distribution.png)
```

### 3. SVM Confusion Matrix

```markdown
![SVM Confusion Matrix](images/svm_confusion_matrix.png)
```

### 4. Decision Tree Confusion Matrix

```markdown
![Decision Tree Confusion Matrix](images/dt_confusion_matrix.png)
```

---

**Note**: Adjust paths if you're running the script locally or on a different platform than Google Colab. Ensure required images are saved and uploaded in your repository to make visualizations display correctly.
