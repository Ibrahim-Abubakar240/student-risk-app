import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Student Risk Prediction System", layout="wide")
st.title("🎓 Student Risk Prediction System")

@st.cache_data
def load_data():
    df = pd.read_csv("Student_Performance.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("logistic_regression_model.pkl")
    return model

df = load_data()
model = load_model()

# Features = all numeric except risk_status
features = [c for c in df.columns if c not in ['risk_status', 'Risk', 'at_risk', 'StudentID']]

tab1, tab2, tab3 = st.tabs(["🔮 What-If Scenario", "📊 Data Exploration", "📈 Model Performance"])

with tab1:
    st.header("What-If Scenario")
    st.write("Select a student and predict risk with explanation")

    student_id = st.number_input("Enter Student ID / Row Number", min_value=0, max_value=len(df)-1, value=0)
    student_data = df.iloc[student_id:student_id+1]
    X_single = student_data[features]

    st.dataframe(X_single)

    if st.button("Predict Risk", type="primary"):
        pred = model.predict(X_single)[0]
        proba = model.predict_proba(X_single)[0]
        confidence = max(proba) * 100

        if pred == 1:
            st.error(f"🚨 At Risk | Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ Safe | Confidence: {confidence:.1f}%")

        st.divider()
        st.subheader("Explainability - Why this prediction? (SHAP)")

        try:
            # For Logistic Regression - must use LinearExplainer
            X_background = df[features].sample(50) if len(df) > 50 else df[features]
            explainer = shap.LinearExplainer(model, X_background)
            shap_values = explainer.shap_values(X_single)

            if isinstance(shap_values, list):
                shap_vals = shap_values[0][0]
            else:
                shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            shap_df = pd.DataFrame({
                'Feature': features,
                'SHAP Value': shap_vals,
                'Feature Value': X_single.iloc[0].values
            })
            shap_df['Abs'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values('Abs', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['red' if x > 0 else 'green' for x in shap_df['SHAP Value']]
            ax.barh(shap_df['Feature'][::-1], shap_df['SHAP Value'][::-1], color=colors[::-1])
            ax.set_xlabel("SHAP Value (Impact on Risk)")
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(shap_df[['Feature', 'Feature Value', 'SHAP Value']])
            st.info("🔴 Red = increases risk, 🟢 Green = decreases risk")

        except Exception as e:
            st.warning(f"SHAP could not be drawn: {e}")

with tab2:
    st.header("Data Exploration")
    numeric_cols = df.select_dtypes(include='number').columns
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)

with tab3:
    st.header("Model Performance")
    X = df[features]
    y = df['risk_status']
    preds = model.predict(X)

    st.write(f"**Tested on:** {len(df)} unique students")
    st.write(f"**Accuracy:** {model.score(X, y):.4f}")
    st.caption("Note: This number reflects the whole dataset, including rows the model has seen.")

    st.write("**Confusion Matrix**")
    st.dataframe(pd.DataFrame(confusion_matrix(y, preds), columns=['Pred 0', 'Pred 1']))

    st.write("**Classification Report**")
    st.text(classification_report(y, preds))

    st.divider()
    st.subheader("Global Explainability - What drives risk for ALL students? (SHAP)")

    try:
        X_sample = X.sample(100) if len(X) > 100 else X
        explainer_global = shap.LinearExplainer(model, X_sample)
        shap_values_global = explainer_global.shap_values(X_sample)

        st.write("Top features overall:")
        fig_bar, ax_bar = plt.subplots()
        shap.summary_plot(shap_values_global, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig_bar)

        st.write("Detailed distribution:")
        fig_bee, ax_bee = plt.subplots()
        shap.summary_plot(shap_values_global, X_sample, show=False)
        plt.tight_layout()
        st.pyplot(fig_bee)

    except Exception as e:
        st.write(f"Global SHAP error: {e}")
        if hasattr(model, 'coef_'):
            imp = pd.DataFrame({'Feature': features, 'Weight': model.coef_[0]}).sort_values('Weight', key=abs, ascending=False)
            st.bar_chart(imp.set_index('Feature'))
