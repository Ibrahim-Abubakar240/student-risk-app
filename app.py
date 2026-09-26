import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Student Risk Prediction System", layout="wide")
model = joblib.load('logistic_regression_model.pkl')
st.title("🎓 Student Risk Prediction System")

@st.cache_data
def load_data():
    df = pd.read_csv("Student_Performance.csv")
    rows_before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first").reset_index(drop=True)
    rows_after = len(df)
    grade_bins = [-np.inf, 40, 45, 50, 60, 70, 101]
    grade_labels = ['f', 'e', 'd', 'c', 'b', 'a']
    df['final_grade'] = pd.cut(df['overall_score'], bins=grade_bins, labels=grade_labels, right=False).astype(str)
    df['risk_status'] = df['final_grade'].str.lower().map({'a':1,'b':1,'c':1,'d':0,'e':0,'f':0})
    return df, rows_before, rows_after, rows_before-rows_after

df, rows_before, rows_after, dup_removed = load_data()
features = ["age","gender","school_type","parent_education","study_hours","attendance_percentage","internet_access","travel_time","extra_activities","study_method"]

# --- CORE: REASONS FROM YOUR DATASET ---
def get_reasons_and_recommendation(row, pred):
    reasons_at_risk = []
    reasons_safe = []

    att = float(row['attendance_percentage'])
    sh = float(row['study_hours'])

    if att < 70: reasons_at_risk.append(f"Low Attendance: Only {att:.0f}% attendance. Students need at least 75% to pass. Missing classes = missing lessons.")
    else: reasons_safe.append(f"Good Attendance: {att:.0f}% - Regular in class.")

    if sh < 2: reasons_at_risk.append(f"Very Low Study Hours: {sh} hrs/day at home. This is not enough to understand subjects.")
    elif sh < 3: reasons_at_risk.append(f"Low Study Hours: {sh} hrs/day. Should be 3-4 hours.")
    else: reasons_safe.append(f"Good Study Hours: {sh} hrs/day.")

    if str(row['internet_access']).lower() == 'no': reasons_at_risk.append("No Internet Access: Cannot research or do online homework.")
    else: reasons_safe.append("Has Internet: Can learn online.")

    if str(row['travel_time']) in ['30-60 min', '>60 min']: reasons_at_risk.append(f"Long Travel Time: {row['travel_time']} to school. Student gets tired.")

    if str(row['extra_activities']).lower() == 'no': reasons_at_risk.append("No Extra Activities: Not in clubs/sports, may affect interest in school.")
    else: reasons_safe.append("Active in Extra Activities: Good for motivation.")

    if str(row['parent_education']).lower() in ['no formal','high school']: reasons_at_risk.append(f"Parent Education: {row['parent_education']} - Parents may need guidance on supporting study at home.")

    # Recommendations
    if pred == 0: # At Risk
        rec = ["**👨‍🏫 Recommendation for Teacher/Parent:**"]
        if att < 75: rec.append("1. Call parents this week about attendance. Find why student misses school.")
        if sh < 3: rec.append("2. Make a simple study timetable (2 hrs evening) and check daily.")
        if str(row['internet_access']).lower() == 'no': rec.append("3. Give printed notes, allow library after school.")
        rec.append("4. Put student in extra lesson group and monitor for 2 weeks.")
    else: # Not At Risk
        rec = ["**✅ Recommendation:**", "1. Student is doing well. Keep encouraging.", "2. Let student help struggling classmates.", "3. Continue monitoring monthly."]

    return reasons_at_risk, reasons_safe, rec

# --- SIDEBAR ---
def options_for(col, fallback):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else fallback

st.sidebar.header("🔍 Check Student by ID")
sid_input = st.sidebar.text_input("Enter Student ID")

if st.sidebar.button("Load Student"):
    try:
        sid = int(sid_input)
        if sid not in df['student_id'].values:
            st.sidebar.error(f"ID {sid} not found")
        else:
            st.session_state['sid'] = sid
    except: st.sidebar.error("Enter a number")

if 'sid' in st.session_state:
    row = df[df['student_id'] == st.session_state['sid']].iloc[0]
    X = pd.DataFrame([row[features]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf = proba[0]*100 if pred==0 else proba[1]*100

    st.subheader(f"Student ID: {row['student_id']} | Grade: {row['final_grade'].upper()} | Score: {row['overall_score']}")

    if pred == 0:
        st.error(f"### 🚨 AT RISK - Model Confidence: {conf:.0f}%")
    else:
        st.success(f"### ✅ NOT AT RISK - Model Confidence: {conf:.0f}%")

    at_risk_reasons, safe_reasons, recommendations = get_reasons_and_recommendation(row, pred)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 🔴 Why At Risk? (Problems Found)")
        if at_risk_reasons:
            for r in at_risk_reasons: st.write(f"- {r}")
        else: st.write("No major problems found.")

    with col2:
        st.write("#### 🟢 Why Safe? (Good Habits)")
        if safe_reasons:
            for r in safe_reasons: st.write(f"- {r}")
        else: st.write("Few good habits recorded.")

    st.divider()
    for rec in recommendations: st.write(rec)

    st.divider()
    st.write("**Full Record from Dataset:**")
    st.dataframe(row.to_frame().T)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Visuals", "🤖 Performance"])
with tab1: st.dataframe(df.head(20))
with tab2:
    fig = plt.figure(); sns.countplot(x='risk_status', data=df); plt.xticks([0,1],['At Risk','Not At Risk']); st.pyplot(fig)
    fig = plt.figure(); sns.boxplot(x='risk_status', y='overall_score', data=df); plt.xticks([0,1],['At Risk','Not At Risk']); st.pyplot(fig)
with tab3:
    X=df[features]; y=df['risk_status']; preds=model.predict(X)
    st.write(f"Accuracy: {model.score(X,y):.4f}"); st.dataframe(pd.DataFrame(confusion_matrix(y,preds))); st.text(classification_report(y,preds))
