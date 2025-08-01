# Forest-cover-type-classification


## 📌 Overview

This project implements a **multi-class classification model** to predict the **forest cover type** using cartographic and environmental features from the **Covertype dataset (UCI)**. The objective is to identify the type of vegetation cover based on geographical and soil-related attributes using tree-based machine learning models such as **Random Forest** and **XGBoost**.

---

## 🗂️ Dataset

- **Source:** [UCI Covertype Dataset](https://archive.ics.uci.edu/ml/datasets/covertype)  
- **Download:** [covtype.data.gz (direct)](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz)
- **Records:** 581,012 samples
- **Features:**
  - 10 numerical features (e.g., `Elevation`, `Slope`, `Aspect`)
  - 44 binary categorical features:
    - 4 for `Wilderness_Area`
    - 40 for `Soil_Type`
  - 1 target column: `Cover_Type` (values 1 to 7, mapped to 0 to 6 for modeling)

---

## 🎯 Objectives

- Load and preprocess the dataset (including outlier handling and scaling)
- Prepare features and transform the target for multi-class learning
- Train and evaluate classification models (Random Forest, XGBoost)
- Compare model performances using accuracy and F1-score
- Visualize the confusion matrix and feature importances
- *(Bonus)* Experiment with additional models or hyperparameter tuning

---

## 🛠️ Tools & Libraries

- **Python**
- **Pandas / NumPy** – data manipulation
- **Matplotlib / Seaborn** – visualizations
- **Scikit-learn** – preprocessing, evaluation, and modeling
- **XGBoost** – high-performance tree-based modeling

---

## 📊 Key Components

### 1. 📥 Data Preprocessing

- Loaded and labeled dataset using provided UCI documentation
- Checked for missing values and column data types
- Capped outliers in numeric features using IQR or Z-score
- Converted target labels from `1–7` to `0–6` to match `XGBoost` requirements

### 2. ✂️ Feature Handling

- Used all features initially (no dimensionality reduction)
- Ensured binary categorical features (`Wilderness_Area`, `Soil_Type`) were left as-is
- Scaled numerical features (optional depending on model)

### 3. 🔁 Train/Test Split

- Split the data using `train_test_split()` with stratification
- Preserved class balance across training and test sets

### 4. 🤖 Modeling

- Trained a **Random Forest** as a baseline model
- Trained an **XGBoost Classifier** with:
  - Objective: `multi:softmax`
  - Evaluation metric: `mlogloss`
  - Number of classes: 7
- Compared performance using accuracy and classification report

### 5. 📈 Evaluation

- Evaluated models using:
  - **Accuracy**
  - **Precision, Recall, F1-score**
  - **Confusion Matrix**
- Plotted confusion matrix for visual understanding of misclassifications

### 6. 🔍 Feature Importance

- Extracted and visualized feature importances from XGBoost
- Analyzed top contributing features to model predictions

---

## 🚀 Bonus Work

- Compared **Random Forest** and **XGBoost** performance
- Tried varying `max_depth`, `learning_rate`, and `n_estimators` in XGBoost
- *(Optional)* Used GridSearchCV or RandomizedSearchCV for tuning

---

## ✅ Covered Topics

- Supervised Learning (Multi-class Classification)
- Tree-Based Models (Random Forest, XGBoost)
- Data Preprocessing & Outlier Handling
- Evaluation Metrics (Accuracy, F1-score)
- Feature Importance Analysis
- Confusion Matrix Visualization

---

## 🏁 Future Improvements

- Use **feature selection** to reduce model complexity
- Apply **dimensionality reduction** (e.g., PCA) for exploratory visualization
- Try other models like **LightGBM**, **CatBoost**
- Build an interactive **dashboard** using Streamlit to demo predictions

