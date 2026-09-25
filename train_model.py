import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv("Student_Performance.csv")

# Remove duplicate students, same as app.py
df = df.drop_duplicates(subset="student_id", keep="first").reset_index(drop=True)

# Recompute final_grade from overall_score (same bands as app.py)
grade_bins = [-np.inf, 40, 45, 50, 60, 70, 101]
grade_labels = ['f', 'e', 'd', 'c', 'b', 'a']
df['final_grade'] = pd.cut(df['overall_score'], bins=grade_bins, labels=grade_labels, right=False).astype(str)

# Derive risk_status same way as app.py: 1 = Not At Risk (a,b,c), 0 = At Risk (d,e,f)
df['risk_status'] = df['final_grade'].str.lower().map({
    'a': 1, 'b': 1, 'c': 1,
    'd': 0, 'e': 0, 'f': 0
})

# Features — must match app.py's `features` list exactly (scores excluded on purpose)
features = [
    "age", "gender", "school_type", "parent_education", "study_hours",
    "attendance_percentage", "internet_access", "travel_time", "extra_activities",
    "study_method"
]

X = df[features]
y = df['risk_status']

# Identify numeric and categorical columns among the selected features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Build pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved successfully as logistic_regression_model.pkl")