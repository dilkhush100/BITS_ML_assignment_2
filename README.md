# Heart Failure Prediction - Clinical Records (Classification)

## a. Problem Statement
The objective of this assignment is to develop an end-to-end machine learning pipeline that predicts patient mortality (survival vs. death) based on clinical medical records. By training and evaluating multiple classification algorithms on clinical features, the goal is to identify the most robust model for medical diagnostic prediction, ultimately assisting in early detection and proactive healthcare intervention.

## b. Dataset Description
The model utilizes the **Heart Failure Prediction - Clinical Records** dataset sourced from Kaggle. 
- **Total Instances:** 5,000 patient records.
- **Features:** 12 predictor clinical attributes (including age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, and time).
- **Target Variable:** `DEATH_EVENT` (Binary: 1 for deceased, 0 for survived).
- **Preprocessing:** The dataset was partitioned using a stratified 80/20 train-test split to maintain class distributions. Continuous features were scaled using `StandardScaler` to ensure optimal performance for distance-based algorithms.

## c. Github Repository Link
https://github.com/dilkhush100/BITS_ML_assignment_2

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8300 | 0.9006 | 0.7416 | 0.7038 | 0.7222 | 0.6003 |
| Decision Tree | 0.9240 | 0.9612 | 0.9190 | 0.8312 | 0.8729 | 0.8210 |
| kNN | 0.9790 | 0.9793 | 0.9803 | 0.9522 | 0.9661 | 0.9511 |
| Naive Bayes | 0.7900 | 0.8848 | 0.7364 | 0.5159 | 0.6067 | 0.4833 |
| Random Forest (Ensemble) | 0.9910 | 0.9999 | 0.9967 | 0.9745 | 0.9855 | 0.9791 |

### Observations about model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Provided a strong linear baseline. The high AUC indicates well-calibrated probabilities, but it slightly underperformed tree-based methods due to its inability to capture non-linear interactions in clinical thresholds (e.g., age combined with ejection fraction). |
| Decision Tree | Demonstrated excellent non-linear mapping capabilities, achieving high accuracy (92.4%) and AUC (0.9612), effectively capturing complex clinical thresholds without severe overfitting. |
| kNN | Benefited significantly from the `StandardScaler` and achieved exceptionally high performance across all metrics (Accuracy: 97.9%), proving that patients with similar mortality risks are tightly clustered within this specific clinical feature space. |
| Naive Bayes | Yielded the lowest recall and MCC. This is expected, as clinical features (like smoking, blood pressure, and heart functionality) are highly correlated, violating the algorithm's core assumption of feature independence. |
| Random Forest (Ensemble) | Solved the single Decision Tree's variance issues through bootstrapping. It generalized exceptionally well on the unseen test data, exhibiting the highest precision and F1 score. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)**. It dominated across all 6 required metrics, specifically achieving the highest MCC (Matthews Correlation Coefficient) of 0.9791. This makes it the most robust choice for a medical diagnostic tool where minimizing both false positives and false negatives carries significant clinical weight. |
