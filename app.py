import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Configure the Streamlit page layout
st.set_page_config(
    page_title="Heart Failure ML Dashboard",
    page_icon="🫀",
    layout="wide"
)

# Map UI dropdown options exactly to the saved .pkl filenames from Stage 3
MODEL_DICTIONARY = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'kNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl'
}

@st.cache_resource
def load_artifacts(model_name):
    """Load the scaler and selected model efficiently using Streamlit caching."""
    scaler_obj = joblib.load('model/scaler.pkl')
    model_obj = joblib.load(f'model/{MODEL_DICTIONARY[model_name]}')
    return scaler_obj, model_obj

def main():
    st.title("🫀 Heart Failure Prediction - Clinical Evaluation Pipeline")
    st.markdown("""
    **BITS WILP M.Tech Assignment 2**  
    Upload unseen clinical test data below to evaluate the performance of the trained classification models.
    """)
    
    st.sidebar.header("Pipeline Configuration")
    
    # 1. CSV test-data upload
    uploaded_csv = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])
    
    # 2. Model-selection dropdown
    selected_model = st.sidebar.selectbox("Select Classification Algorithm", list(MODEL_DICTIONARY.keys()))
    
    # Separator in sidebar
    st.sidebar.divider()
    
    if uploaded_csv is not None:
        try:
            # Load the uploaded dataset
            df_test = pd.read_csv(uploaded_csv)
            st.sidebar.success("✅ Test dataset successfully loaded!")
            
            target_col = 'DEATH_EVENT'
            if target_col not in df_test.columns:
                st.error(f"Dataset Validation Error: The required target column '{target_col}' was not found.")
                return
            
            # Separate features and target
            X_test_raw = df_test.drop(columns=[target_col])
            y_test_true = df_test[target_col]
            
            # 3. Load artifacts and prevent retraining
            scaler, model = load_artifacts(selected_model)
            
            # Apply preprocessing (Scaling)
            X_test_scaled = scaler.transform(X_test_raw)
            
            if st.sidebar.button(f"Evaluate {selected_model}"):
                
                # Execute Predictions
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # 6. Clearly show the selected model
                st.markdown(f"### 📊 Evaluation Dashboard: **{selected_model}**")
                
                # 4. Display the 6 required metrics side-by-side
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_test_true, y_pred):.4f}")
                m2.metric("AUC", f"{roc_auc_score(y_test_true, y_proba):.4f}")
                m3.metric("Precision", f"{precision_score(y_test_true, y_pred, zero_division=0):.4f}")
                m4.metric("Recall", f"{recall_score(y_test_true, y_pred):.4f}")
                m5.metric("F1 Score", f"{f1_score(y_test_true, y_pred):.4f}")
                m6.metric("MCC", f"{matthews_corrcoef(y_test_true, y_pred):.4f}")
                
                st.divider()
                
                # 5. Display Confusion Matrix AND Classification Report
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test_true, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Survived (0)', 'Deceased (1)'],
                                yticklabels=['Survived (0)', 'Deceased (1)'])
                    ax.set_ylabel('Actual Clinical Outcome')
                    ax.set_xlabel('Model Predicted Outcome')
                    st.pyplot(fig)
                
                with col_right:
                    st.subheader("Classification Report")
                    report = classification_report(y_test_true, y_pred, zero_division=0)
                    st.code(report, language="text")
                    
        except Exception as e:
            st.error(f"An execution error occurred: {e}. Please ensure you are uploading the correct 'test_data.csv' and that all .pkl files are in the 'model/' directory.")
            
    else:
        st.info("👈 Please upload the 'test_data.csv' file via the Configuration Panel to begin.")

if __name__ == "__main__":
    main()