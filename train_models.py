import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
import joblib

# Importing the 5 required classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. Load Data
    # Ensure the downloaded Kaggle CSV is in the same directory
    file_path = "heart_failure_clinical_records.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please place it in the root folder.")
        return
        
    df = pd.read_csv(file_path)

    # 2. Separate Features and Target
    target_col = 'DEATH_EVENT'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Stratified Train/Test Split
    # Stratification ensures both sets have the same proportion of survival/death cases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Export test dataset (features + target) for the Streamlit app requirement
    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_df.to_csv("test_data.csv", index=False)
    print("-> Successfully exported 'test_data.csv' for Streamlit inference.")

    # 4. Preprocessing: Feature Scaling
    # To prevent data leakage, fit the scaler ONLY on the training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the fitted scaler artifact
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")

    # 5. Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # 6. Train, Evaluate, and Save Artifacts
    print("\n--- Model Evaluation Metrics (Test Set) ---")
    header = f"{'ML Model Name':<25} | {'Accuracy':<8} | {'AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'MCC':<8}"
    print(header)
    print("-" * len(header))

    for name, clf in models.items():
        # Train
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Print formatted row for easy copying to README.md
        print(f"{name:<25} | {acc:.4f}   | {auc:.4f}   | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {mcc:.4f}")

        # Serialize and save the trained model
        safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(clf, f"model/{safe_filename}.pkl")

    print("\n-> All models trained and saved to the 'model/' directory successfully.")

if __name__ == "__main__":
    main()
