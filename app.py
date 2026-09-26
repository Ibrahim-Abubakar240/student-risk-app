import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import shap

st.set_page_config(page_title="Student Risk Prediction System", layout="wide")

# Load Model
model = joblib.load('logistic_regression_model.pkl')

st.title("🎓 Student Risk Prediction System")

# ---------------------------------------------------------------------
# LOAD DATASET — mirrors the training script exactly:
#   - dedupe by student_id (keep first)
#   - recompute final_grade from overall_score using the same bands
#   - derive Risk the same way (a,b,c = Not At Risk / d,e,f = At Risk)
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Student_Performance.csv")

    rows_before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first").reset_index(drop=True)
    rows_after = len(df)
    duplicates_removed = rows_before - rows_after

    # A: 70-100  B: 60-69  C: 50-59  D: 45-49  E: 40-44  F: below 40
    grade_bins = [-np.inf, 40, 45, 50, 60, 70, 101]
    grade_labels = ['f', 'e', 'd', 'c', 'b', 'a']
    df['final_grade'] = pd.cut(df['overall_score'], bins=grade_bins, labels=grade_labels, right=False).astype(str)

    # risk_status: 1 = Not At Risk (a,b,c), 0 = At Risk (d,e,f) — matches the "Risk" target in training
    df['risk_status'] = df['final_grade'].str.lower().map({
        'a': 1, 'b': 1, 'c': 1,
        'd': 0, 'e': 0, 'f': 0
    })

    return df, rows_before, rows_after, duplicates_removed

df, rows_before, rows_after, duplicates_removed = load_data()

st.caption(
    f"Dataset: **{rows_before}** total rows → **{duplicates_removed}** duplicate student_id row(s) removed "
    f"→ **{rows_after}** unique students used for training/testing below."
)

# FEATURES THE MODEL NEEDS — must match the training script's `features` list exactly.
# math_score, science_score, english_score, and overall_score are deliberately
# excluded since Risk is derived from overall_score; including them would let
# the model just read the answer off one input instead of learning genuine
# early-warning patterns from behavior/demographics.
features = [
    "age", "gender", "school_type", "parent_education", "study_hours",
    "attendance_percentage", "internet_access", "travel_time", "extra_activities",
    "study_method"
]

# Build dropdown options straight from the dataset so they always match
# whatever categories the model's encoder was actually trained on.
def options_for(col, fallback):
    if col in df.columns:
        vals = sorted(df[col].dropna().unique().tolist())
        if vals:
            return vals
    return fallback

gender_options = options_for('gender', ['male', 'female', 'other'])
school_type_options = options_for('school_type', ['public', 'private'])
parent_education_options = options_for('parent_education', ['no formal', 'high school', 'diploma', 'graduate', 'phd'])
internet_access_options = options_for('internet_access', ['yes', 'no'])
travel_time_options = options_for('travel_time', ['<15 min', '15-30 min', '30-60 min', '>60 min'])
extra_activities_options = options_for('extra_activities', ['yes', 'no'])
study_method_options = options_for('study_method', ['notes', 'textbook', 'coaching', 'group study', 'online video', 'mixed'])

# ---------------------------------------------------------------------
# DEFAULT VALUES FOR THE SIDEBAR WIDGETS (only used before anything's loaded)
# ---------------------------------------------------------------------
defaults = {
    'age': 16,
    'gender': gender_options[0],
    'school_type': school_type_options[0],
    'parent_education': parent_education_options[0],
    'study_hours': 3.0,
    'attendance_percentage': 70.0,
    'internet_access': internet_access_options[0],
    'travel_time': travel_time_options[0],
    'extra_activities': extra_activities_options[0],
    'study_method': study_method_options[0],
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

st.session_state.setdefault('loaded_student_id', None)
st.session_state.setdefault('load_message', None)

# ---------------------------------------------------------------------
# SIDEBAR — CHECK STUDENT BY ID (loads real values into the form below)
# ---------------------------------------------------------------------
st.sidebar.header("🔍 Check Student by ID")
st.sidebar.text_input("Enter Student ID", key="student_id_input")

def load_student():
    raw_id = st.session_state.student_id_input.strip()
    if raw_id == "":
        st.session_state.load_message = ("error", "Please enter a Student ID.")
        return
    try:
        sid = int(raw_id)
    except ValueError:
        st.session_state.load_message = ("error", "Student ID must be a number.")
        return

    match = df[df['student_id'] == sid]
    if match.empty:
        st.session_state.load_message = ("error", f"❌ Student ID {sid} not found.")
        return

    row = match.iloc[0]
    st.session_state.age = int(row['age'])
    st.session_state.gender = row['gender']
    st.session_state.school_type = row['school_type']
    st.session_state.parent_education = row['parent_education']
    st.session_state.study_hours = float(row['study_hours'])
    st.session_state.attendance_percentage = float(row['attendance_percentage'])
    st.session_state.internet_access = row['internet_access']
    st.session_state.travel_time = row['travel_time']
    st.session_state.extra_activities = row['extra_activities']
    st.session_state.study_method = row['study_method']

    st.session_state.loaded_student_id = sid
    actual_risk = "Not At Risk ✅" if row['risk_status'] == 1 else "At Risk ⚠️"
    st.session_state.load_message = (
        "success",
        f"Loaded Student ID {sid} — Final Grade: {row['final_grade'].upper()} — Actual Risk Status: {actual_risk}"
    )

st.sidebar.button("Load Student", on_click=load_student)

if st.session_state.load_message:
    level, msg = st.session_state.load_message
    getattr(st.sidebar, level)(msg)

st.sidebar.markdown("---")

# ---------------------------------------------------------------------
# SIDEBAR — PREDICTION FORM (prefilled by "Load Student", editable)
# ---------------------------------------------------------------------
st.sidebar.header("📊 Predict Risk")
st.sidebar.caption("Loading a student above fills this in with their real data. Adjust any field to test a what-if scenario.")

st.sidebar.slider("Age", 14, 19, key='age')
st.sidebar.selectbox("Gender", gender_options, key='gender')
st.sidebar.selectbox("School Type", school_type_options, key='school_type')
st.sidebar.selectbox("Parent Education", parent_education_options, key='parent_education')
st.sidebar.slider("Study Hours", 0.0, 10.0, key='study_hours')
st.sidebar.slider("Attendance %", 0.0, 100.0, key='attendance_percentage')
st.sidebar.selectbox("Internet Access", internet_access_options, key='internet_access')
st.sidebar.selectbox("Travel Time", travel_time_options, key='travel_time')
st.sidebar.selectbox("Extra Activities", extra_activities_options, key='extra_activities')
st.sidebar.selectbox("Study Method", study_method_options, key='study_method')

if st.sidebar.button("Predict Risk"):
    input_data = pd.DataFrame([[
        st.session_state.age, st.session_state.gender, st.session_state.school_type,
        st.session_state.parent_education, st.session_state.study_hours,
        st.session_state.attendance_percentage, st.session_state.internet_access,
        st.session_state.travel_time, st.session_state.extra_activities,
        st.session_state.study_method
    ]], columns=features)

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    if pred == 1:
        st.sidebar.success(f"✅ Not At Risk | Confidence: {proba[1]*100:.1f}%")
    else:
        st.sidebar.error(f"⚠️ At Risk | Confidence: {proba[0]*100:.1f}%")

    # Store this prediction's input so the main area can show the SHAP explanation
    st.session_state['last_prediction_input'] = input_data
    st.session_state['last_prediction_label'] = pred
    st.session_state['last_prediction_proba'] = proba

# ---------------------------------------------------------------------
# MAIN AREA — show full record of the loaded student, if any
# ---------------------------------------------------------------------
if st.session_state.loaded_student_id is not None:
    sid = st.session_state.loaded_student_id
    student = df[df['student_id'] == sid]
    if not student.empty:
        st.subheader(f"Student ID: {sid}")
        st.write(f"**Final Grade:** {student['final_grade'].values[0].upper()}")
        risk = "Not At Risk ✅" if student['risk_status'].values[0] == 1 else "At Risk ⚠️"
        st.write(f"**Risk Status:** {risk}")
        st.dataframe(student)

# ---------------------------------------------------------------------
# SHAP EXPLANATION — shows which factors drove the most recent prediction
# ---------------------------------------------------------------------
if 'last_prediction_input' in st.session_state:
    st.subheader("🔎 Why did the model make this prediction?")

    with st.spinner("Calculating feature contributions..."):
        # Background sample: a small random subset of real students,
        # used as a reference point for SHAP to measure each feature's impact against.
        background = df[features].sample(n=100, random_state=42)

        # A model-agnostic explainer wrapping the full pipeline's predict_proba,
        # so explanations are given in terms of the original 10 features,
        # not the internally expanded one-hot encoded columns.
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(st.session_state['last_prediction_input'])

    # Class index: 0 = At Risk, 1 = Not At Risk — explain whichever class was predicted
    pred_class = st.session_state['last_prediction_label']
    values = shap_values.values[0, :, pred_class]
    feature_names = st.session_state['last_prediction_input'].columns.tolist()

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": values
    }).sort_values(by="Contribution", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in contrib_df["Contribution"]]
    ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors)
    ax.set_xlabel("Contribution to Prediction")
    ax.set_title(
        f"Feature Contributions to '{'Not At Risk' if pred_class == 1 else 'At Risk'}' Prediction"
    )
    st.pyplot(fig)

    st.caption(
        "Green bars pushed the prediction toward the shown outcome; red bars pushed against it. "
        "Longer bars indicate a stronger influence on this specific student's prediction."
    )

# MAIN TABS
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Visualizations", "🤖 Model Performance"])

with tab1:
    st.header("Dataset Preview")
    st.metric("Unique Students (after removing duplicates)", rows_after)
    st.dataframe(df.head(10))

    st.header("Missing Values Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("Descriptive Statistics")
    st.dataframe(df.describe())

    st.header("Risk Distribution")
    fig = plt.figure()
    sns.countplot(x='risk_status', data=df)
    plt.xticks([0, 1], ['At Risk', 'Not At Risk'])
    st.pyplot(fig)

    st.header("Overall Score vs Risk")
    fig = plt.figure()
    sns.boxplot(x='risk_status', y='overall_score', data=df)
    plt.xticks([0, 1], ['At Risk', 'Not At Risk'])
    st.pyplot(fig)

    st.header("Correlation Matrix")
    # This is for exploration only, so it includes ALL numeric columns
    # (including scores), independent of what the model actually uses to predict.
    numeric_cols = df.select_dtypes(include='number').columns
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)

with tab3:
    st.header("Model Performance")
    X = df[features]
    y = df['risk_status']
    preds = model.predict(X)

    st.write(f"**Tested on:** {len(df)} unique students (duplicates already removed)")
    st.write(f"**Accuracy:** {model.score(X, y):.4f}")
    st.caption("Note: this number reflects the whole dataset, including rows the model was trained on, so it will read higher than the true test accuracy printed by the training script.")
    st.write("**Confusion Matrix**")
    st.dataframe(pd.DataFrame(confusion_matrix(y, preds), columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']))

    st.write("**Classification Report**")
    st.text(classification_report(y, preds))
