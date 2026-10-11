"""Heart Disease Prediction System: Streamlit web app (Assignment 2, Task 2).

Course : AIC354 Machine Learning Fundamentals, COMSATS University Islamabad, Lahore Campus
Student: Haziq Mahmood (FA24-BSE-049)

The app loads the trained scikit-learn pipeline (StandardScaler + SVC) that was saved by
Heart_Disease_Prediction.ipynb and predicts, from 4 inputs, whether a patient has heart
disease (output 1) or not (output 0).
"""
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"
DATA_PATH = BASE_DIR / "processed_cleveland.csv"
INFO_PATH = BASE_DIR / "model_info.json"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

# The 4 input attributes, in the order the model was trained with
FEATURES = ["cp", "thalach", "oldpeak", "exang"]

# Range of values of each input (identical to Step 4.5 of the notebook)
CP_LABELS = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
THALACH_RANGE = (71, 202)       # beats per minute
OLDPEAK_RANGE = (0.0, 6.2)      # ST depression


def train_fallback_model():
    """Re-create the notebook's model from the CSV (same split, same hyperparameters).

    Only used if the saved .pkl file cannot be loaded, for example because the cloud
    machine has a different scikit-learn version than the one used for training.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    data = pd.read_csv(DATA_PATH, na_values="?")
    data["target"] = (data["num"] > 0).astype(int)
    train_df, _ = train_test_split(data, test_size=0.20, stratify=data["target"], random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("model", SVC(C=0.5, gamma=0.01, kernel="rbf"))])
    return pipeline.fit(train_df[FEATURES], train_df["target"])


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved model (no re-training); fall back to re-creating it if loading fails."""
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return train_fallback_model()


@st.cache_data(show_spinner=False)
def load_info():
    try:
        return json.loads(INFO_PATH.read_text())
    except Exception:
        return None


def save_feedback(rating, comment, inputs, prediction):
    """Append one feedback row to feedback.csv. Returns True on success."""
    try:
        is_new = not FEEDBACK_PATH.exists()
        with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "rating", "comment", *FEATURES, "prediction"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), rating,
                             comment.strip(), *[inputs[k] for k in FEATURES], prediction])
        return True
    except OSError:
        return False


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

model = load_model()
info = load_info()

st.title("Heart Disease Prediction System")
st.write(
    "Enter four values from a patient's chest-pain assessment and exercise stress test. "
    "The model predicts **0 (no heart disease)** or **1 (heart disease present)**."
)
st.warning(
    "Educational project for AIC354 Machine Learning Fundamentals. "
    "This is **not** a medical device and must not be used for diagnosis. "
    "Always consult a medical professional."
)

with st.form("patient_form"):
    st.subheader("Patient inputs")
    cp = st.selectbox(
        "Chest pain type (cp)", options=list(CP_LABELS), index=2,
        format_func=lambda code: f"{code} - {CP_LABELS[code]}")
    thalach = st.number_input(
        f"Maximum heart rate achieved, bpm (thalach)  [range {THALACH_RANGE[0]} to {THALACH_RANGE[1]}]",
        min_value=THALACH_RANGE[0], max_value=THALACH_RANGE[1], value=150, step=1)
    oldpeak = st.number_input(
        f"ST depression induced by exercise (oldpeak)  [range {OLDPEAK_RANGE[0]} to {OLDPEAK_RANGE[1]}]",
        min_value=OLDPEAK_RANGE[0], max_value=OLDPEAK_RANGE[1], value=1.0, step=0.1, format="%.1f")
    exang = st.radio(
        "Exercise-induced angina (exang)", options=[0, 1], horizontal=True,
        format_func=lambda v: "Yes" if v == 1 else "No")
    submitted = st.form_submit_button("Predict")

if submitted:
    patient = pd.DataFrame([[cp, thalach, oldpeak, exang]], columns=FEATURES).astype(float)
    st.session_state["result"] = {
        "inputs": {"cp": cp, "thalach": thalach, "oldpeak": oldpeak, "exang": exang},
        "prediction": int(model.predict(patient)[0]),
        "score": float(model.decision_function(patient)[0]),
        "feedback_sent": False,
    }

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Result")
    if result["prediction"] == 1:
        st.error("Prediction: **1**, heart disease likely")
    else:
        st.success("Prediction: **0**, no heart disease likely")

    col_class, col_score = st.columns(2)
    col_class.metric("Predicted output (target)", result["prediction"])
    col_score.metric("SVM decision score", f"{result['score']:+.2f}")
    st.caption(
        "The decision score is the signed distance from the SVM decision boundary: positive means "
        "class 1 (disease), negative means class 0. A larger magnitude means the patient lies "
        "further from the boundary. It is not a probability."
    )

    entered = result["inputs"]
    st.table(pd.DataFrame({
        "Input": ["Chest pain type (cp)", "Maximum heart rate (thalach)",
                  "ST depression (oldpeak)", "Exercise-induced angina (exang)"],
        "Value": [f"{entered['cp']} - {CP_LABELS[entered['cp']]}", f"{entered['thalach']} bpm",
                  f"{entered['oldpeak']:.1f}", "Yes" if entered["exang"] == 1 else "No"],
    }))

    st.subheader("Feedback")
    if result["feedback_sent"]:
        st.success("Thank you for your feedback!")
    else:
        with st.form("feedback_form", clear_on_submit=True):
            rating = st.radio("Was this result easy to understand?", ["Yes", "Somewhat", "No"], horizontal=True)
            comment = st.text_area("Comments or suggestions (optional)", max_chars=300)
            send = st.form_submit_button("Send feedback")
        if send:
            save_feedback(rating, comment, entered, result["prediction"])
            result["feedback_sent"] = True
            st.success("Thank you for your feedback!")

with st.expander("About this model"):
    st.write(
        "A **Support Vector Classifier** (with feature scaling) trained on the UCI Heart Disease "
        "dataset, Cleveland Clinic Foundation subset (303 patients). It uses 4 of the 13 available "
        "attributes. The target is 1 if the patient has heart disease and 0 otherwise."
    )
    if info:
        st.write(
            f"- Cross-validated accuracy (training set): **{info['cv_accuracy']:.2f}**\n"
            f"- Test accuracy (61 unseen patients): **{info['test_metrics']['Accuracy']:.2f}**\n"
            f"- Test recall (sensitivity): **{info['test_metrics']['Recall (sensitivity)']:.2f}**\n"
            f"- Test ROC-AUC: **{info['test_metrics']['ROC-AUC']:.2f}**"
        )
    st.write(
        "The model was trained only on patients inside the input ranges shown above, so values "
        "outside those ranges are not accepted."
    )
