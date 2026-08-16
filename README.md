# Heart Failure Prediction - Clinical Records (Classification)

## a. Problem Statement
The objective of this assignment is to develop an end-to-end machine learning pipeline that predicts patient mortality (survival vs. death) based on clinical medical records. By training and evaluating multiple classification algorithms on clinical features, the goal is to identify the most robust model for medical diagnostic prediction, ultimately assisting in early detection and proactive healthcare intervention.

## b. Dataset Description
The model utilizes the **Heart Failure Prediction - Clinical Records** dataset. 
- **Dataset Link:** [Kaggle - Heart Failure Prediction Clinical Records](https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records)
- **Total Instances:** 5,000 patient records (Train/Test split of 80/20 yields 1,000 instances in the test set).
- **Features:** 12 predictor clinical attributes (including age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, and time).
- **Target Variable:** `DEATH_EVENT` (Binary: 1 for deceased, 0 for survived).
- **Preprocessing:** The dataset was partitioned using a reproducible stratified 80/20 train-test split to maintain the distribution of the target classes. Continuous features were scaled using `StandardScaler` (fitted strictly on the training set to prevent data leakage) to optimize the performance of distance-based algorithms like kNN.

## c. Github Repository Link
https://github.com/dilkhush100/BITS_ML_assignment_2

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8300 | 0.9006 | 0.7416 | 0.7038 | 0.7222 | 0.6003 |
| Decision Tree | 0.9240 | 0.9612 | 0.9190 | 0.8312 | 0.8729 | 0.8210 |
| kNN | 0.9790 | 0.9791 | 0.9803 | 0.9522 | 0.9661 | 0.9511 |
| Naive Bayes | 0.7900 | 0.8848 | 0.7364 | 0.5159 | 0.6067 | 0.4833 |
| Random Forest (Ensemble) | 0.9910 | 0.9999 | 0.9967 | 0.9745 | 0.9855 | 0.9791 |

### Observations about model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Provided a strong baseline with an 83% accuracy. The high AUC indicates well-calibrated probabilities, but it underperformed tree-based methods due to its inability to capture non-linear interactions in clinical thresholds. |
| Decision Tree | Demonstrated excellent non-linear mapping capabilities, achieving high accuracy (92.4%) and AUC (0.9612), effectively capturing complex clinical thresholds without severe overfitting. |
| kNN | Benefited significantly from the `StandardScaler` and achieved exceptionally high performance across all metrics (Accuracy: 97.9%), proving that patients with similar mortality risks are tightly clustered within this scaled feature space. |
| Naive Bayes | Yielded the lowest recall (51.59%) and MCC (0.4833). This is expected, as clinical features (like smoking, blood pressure, and heart functionality) are highly correlated, heavily violating the algorithm's core assumption of strict feature independence. |
| Random Forest (Ensemble) | Solved the single Decision Tree's variance issues through bootstrapping. It generalized near-perfectly on the unseen test data, exhibiting a 99.1% accuracy and a 0.9855 F1 score. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)**. It dominated across all 6 required metrics, specifically achieving a near-perfect MCC (Matthews Correlation Coefficient) of 0.9791. This makes it the most robust choice for a medical diagnostic tool where minimizing both false positives and false negatives carries significant clinical weight. |