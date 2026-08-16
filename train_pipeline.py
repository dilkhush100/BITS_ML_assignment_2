import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

def run_ml_pipeline():
    print("Starting ML Pipeline...")
    
    # 1 & 2. Load Dataset and Separate Features/Target
    try:
        df = pd.read_csv('heart_failure_clinical_records.csv')
    except FileNotFoundError:
        print("Error: Dataset CSV not found in the root directory.")
        return

    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # 3 & 4. Reproducible Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 15. Create test_data.csv for Streamlit
    test_df = X_test.copy()
    test_df['DEATH_EVENT'] = y_test
    test_df.to_csv('test_data.csv', index=False)
    print("Exported unseen test data to test_data.csv")

    # 5. Prevent Data Leakage (Scale only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 14. Save Preprocessing Components
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')

    # 6. Implement 5 Required Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    print("\nTraining and Evaluating Models...")
    
    # 7 & 8. Train & Evaluate
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # 10. Generate Confusion Matrix & Classification Report
        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"{'='*40}")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # 13. Save Trained Models
        file_name = f"model/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, file_name)

    # 16. Verify Saved Artifacts Can Be Loaded
    print("\nVerifying artifact integrity...")
    try:
        loaded_scaler = joblib.load('model/scaler.pkl')
        loaded_rf = joblib.load('model/random_forest.pkl')
        print("Success: Scaler and models can be loaded without retraining.")
    except Exception as e:
        print(f"Artifact loading failed: {e}")

if __name__ == "__main__":
    run_ml_pipeline()