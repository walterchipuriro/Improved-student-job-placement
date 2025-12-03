import joblib
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3
import shap
import re
import time
import datetime
import json
from lime.lime_tabular import LimeTabularExplainer

from database import init_db, verify_user, register_user
init_db()

# ---- CONFIG ----
st.set_page_config(page_title="Placement Prediction", layout="wide")

# ---- LOAD MODEL, SCALER AND EXPLAINER ----
# model = joblib.load("best_placement_model.pkl")
model = joblib.load("ensemble_placement_model.pkl")
scaler = joblib.load("scaler.pkl")
# explainer = joblib.load("shap_explainer.pkl")


# ---- BACKGROUND ----
def set_background(image_file):
    import base64
    with open(image_file, "rb") as f:
        data = f.read()  # read only once while file is open
        b64 = base64.b64encode(data).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.4);
        z-index: 0;
    }}
    .block-container {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---- SESSION STATE ----
if "page" not in st.session_state:
    st.session_state.page = "home"
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ---- REDIRECT ----
def redirect_with_delay(page_name, delay=3):
    placeholder = st.empty()
    time.sleep(delay)
    st.session_state.page = page_name
    st.rerun()

def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# ---- SQLITE DB FOR PREDICTIONS ----
DB_FILE = "predictions.db"

def init_prediction_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_number TEXT NOT NULL,
            pred_label INTEGER,
            pred_probability REAL,
            result_text TEXT NOT NULL,
            recommendation TEXT,
            features_json TEXT,
            model_version TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---- SAVE AND FETCH PREDICTIONS ----
def save_prediction(reg_number, pred_label, pred_prob, result_text, recommendation, features_dict, model_version="v1"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    import json
    features_json = json.dumps(features_dict)

    c.execute("""
        INSERT INTO predictions 
        (reg_number, pred_label, pred_probability, result_text, recommendation, features_json, model_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        reg_number,
        pred_label,
        pred_prob,
        result_text,
        recommendation,
        features_json,
        model_version,
        created_at
    ))

    conn.commit()
    conn.close()


#---- FETCH ALL PREDICTIONS FOR ADMIN/COUNSELOR ----
def fetch_all_predictions():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT 
            reg_number,
            pred_label,
            pred_probability,
            result_text,
            recommendation,
            features_json,
            model_version,
            created_at
        FROM predictions
        ORDER BY created_at DESC
    """)

    rows = c.fetchall()
    conn.close()
    return rows



# Initialize DB
init_prediction_db()



# -------------------------------------------------------------------
# Helper: reconstruct RAW features DataFrame from predictions table
# -------------------------------------------------------------------
def build_raw_features_from_predictions(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the stored features_json column to reconstruct the original 11 raw features
    for each prediction. This will be used as 'data' for SHAP/LIME.
    """
    rows = []
    for js in df_predictions["features_json"]:
        feat = json.loads(js)

        rows.append({
            "CGPA": float(feat["CGPA"]),
            "Major_Projects": int(feat["Major_Projects"]),
            "Workshops_Certificatios": int(feat["Certifications"]),
            "Mini_Projects": int(feat["Mini_Projects"]),
            "Skills": int(feat["Skills"]),
            "Communication_Skill_Rating": float(feat["Communication"]),
            "Internship": int(feat["Internship"]),
            "Hackathon": int(feat["Hackathon"]),
            "12th_Percentage": 12.0,   # fixed in training
            "10th_Percentage": 10.0,   # fixed in training
            "backlogs": int(feat["Backlogs"]),
        })

    X_raw = pd.DataFrame(rows)
    return X_raw


# -------------------------------------------------------------------
# Helper: add engineered features (match training EXACTLY)
# -------------------------------------------------------------------
def add_engineered_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    From base 11 features, add the 4 engineered ones:
    CGPA_x_10th, CGPA_x_12th, 10th_x_12th, Academic_Avg.
    Return columns in the EXACT same order used during training.
    """
    X = X_raw.copy()

    X["CGPA_x_10th"] = X["CGPA"] * X["10th_Percentage"]
    X["CGPA_x_12th"] = X["CGPA"] * X["12th_Percentage"]
    X["10th_x_12th"] = X["10th_Percentage"] * X["12th_Percentage"]
    X["Academic_Avg"] = (X["CGPA"] + X["10th_Percentage"] + X["12th_Percentage"]) / 3.0

    # enforce correct column order (15 features)
    ordered_cols = [
        "CGPA",
        "Major_Projects",
        "Workshops_Certificatios",
        "Mini_Projects",
        "Skills",
        "Communication_Skill_Rating",
        "Internship",
        "Hackathon",
        "12th_Percentage",
        "10th_Percentage",
        "backlogs",
        "CGPA_x_10th",
        "CGPA_x_12th",
        "10th_x_12th",
        "Academic_Avg",
    ]
    return X[ordered_cols]


# -------------------------------------------------------------------
# Cached SHAP + LIME explainers, using ALL saved predictions as background
# -------------------------------------------------------------------
@st.cache_resource
def get_explainers_and_background(_model, _scaler, df_predictions: pd.DataFrame):

    """
    Builds:
      - SHAP KernelExplainer on scaled 15-D features
      - LIME Tabular explainer on scaled 15-D features
    using all historical predictions as background data.
    """
    if df_predictions.empty:
        return None, None, None, None, None  # nothing to explain yet

    # 1) reconstruct raw + engineered features
    X_raw_all = build_raw_features_from_predictions(df_predictions)
    X_eng_all = add_engineered_features(X_raw_all)

    feature_names = X_eng_all.columns.tolist()

    # 2) scale
    X_scaled_all = scaler.transform(X_eng_all.values)

    # 3) sample background for SHAP (to keep it fast)
    max_bg = min(200, X_scaled_all.shape[0])
    background = X_scaled_all[:max_bg]


    # 4) SHAP explainer on scaled space
    shap_explainer = shap.KernelExplainer(
    lambda X: _model.predict_proba(X)[:, 1],
    background
)

    X_scaled_all = _scaler.transform(X_eng_all.values)


    # LIME explainer on the same scaled space
    lime_explainer = LimeTabularExplainer(
        training_data=X_scaled_all,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        mode="classification"
    )

    return shap_explainer, lime_explainer, X_eng_all, X_scaled_all, feature_names



# ---------------- HOME PAGE ----------------
def home_page():
    set_background("download (6).jpeg")

    # This number controls horizontal shift (positive = right, negative = left)
    shift_value = 200    

    # ------------------ CSS ------------------
    st.markdown(
        f"""
        <style>
        .main-title {{
            font-size: 50px;
            text-align: center;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
            margin-bottom: 20px;
        }}

        .welcome-text {{
            text-align: center;
            font-size: 45px;
            color: #f8f8f8;
            margin-bottom: 60px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }}

        .auth-section {{
            text-align: center;
            margin-top: 100px;
        }}

        /* Shift the center column slightly to the right */
        .home-center {{
            margin-left: {shift_value}px !important;
        }}

        /* Center Streamlit button container */
        div.stButton {{
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }}

        /* Button styling (same as login/register) */
        div.stButton > button {{
            width: 200px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            background: linear-gradient(135deg,#ff7e5f 0%,#feb47b 30%) !important;
            color: white !important;
            border-radius: 20px !important;
            padding: 12px 40px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            margin: 15px !important;
            cursor: pointer !important;
            transition: all 0.3s ease-in-out !important;
            border: none !important;
        }}

        div.stButton > button:hover {{
            background: linear-gradient(135deg,#feb47b 0%,#ff7e5f 100%) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 0 15px rgba(255,255,255,0.6) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ------------------ PAGE TITLES ------------------
    st.markdown('<div class="main-title">Student Job Placement Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-text">Welcome</div>', unsafe_allow_html=True)

    # ------------------ BUTTONS (CENTER + SHIFT) ------------------
    st.markdown('<div class="auth-section">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        # Wrap buttons inside the center-shift container
        st.markdown('<div class="home-center">', unsafe_allow_html=True)

        if st.session_state.logged_in_user:
            st.success(f"Logged in as {st.session_state.logged_in_user} ({st.session_state.role})")

            if st.button("Go to Dashboard"):
                role = st.session_state.role.lower()
                if role == "student":
                    go_to("student")
                elif role == "counselor":
                    go_to("counselor")
                elif role == "admin":
                    go_to("admin")

        else:
            if st.button("Login"):
                go_to("login")

            if st.button("Register"):
                go_to("register")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)




# ---------------- LOGIN PAGE ----------------
def login_page():
    set_background("download (6).jpeg")

    st.markdown("""
        <style>
        # .login-card {
        #     background: rgba(255, 255, 255, 0.1);
        #     backdrop-filter: blur(10px);
        #     border-radius: 20px;
        #     padding: 50px 60px;
        #     box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        #     border: 1px solid rgba(255, 255, 255, 0.2);
        #     max-width: 500px;
        #     width: 100%;
        #     margin: auto;
        # }
        .login-title {
            font-size: 48px;
            text-align: center;
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
            margin-bottom: 40px;
        }
                
        /* Center the Streamlit button container */
        .stButton {
            display: flex !important;
            justify-content: center !important;
            margin-top: 10px !important;
        }

        /* Make button align with input fields */
        .stButton > button {
            width: 100% !important;
            max-width: 450px !important;
            background: linear-gradient(135deg, #FF7E5F 0%, #FEB47B 30%) !important;
            border-radius: 20px !important;
            padding: 12px 40px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            color: white !important;
            cursor: pointer !important;
        }
                
        /* Force BOTH buttons to have identical width */
        div.stButton > button {
            width: 100% !important;
            max-width: 200px !important;   /* SAME SIZE FOR BOTH BUTTONS */
            min-width: 200px !important;   /* LOCK WIDTH */
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
        }
                
        .stTextInput {
            margin-bottom: 30px !important;   /* Increase spacing here */
        }

        </style>
    """, unsafe_allow_html=True)

    # ---------- CENTER EVERYTHING ----------
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        st.markdown('<div class="login-title">Login</div>', unsafe_allow_html=True)

        reg_number = st.text_input(
            "Registration Number / Employee ID",
            placeholder="Enter your ID",
            key="login_reg_number"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        login_clicked = st.button("Login")
        home_clicked = st.button("Back to Home")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- HANDLE BUTTON ACTIONS ----------
    if login_clicked:
        if not reg_number or not password:
            st.warning("Please fill in all fields.")
            return

        user = verify_user(reg_number, password)
        if user:
            _, username, _, role = user
            st.session_state.logged_in_user = username
            st.session_state.role = role
            st.session_state.reg_number = username
            st.success(f"Login successful! Welcome, {username}")

            if role == "student":
                go_to("student")
            elif role == "counselor":
                go_to("counselor")
            elif role == "admin":
                go_to("admin")
        else:
            st.error("Invalid Registration Number or Password.")

    if home_clicked:
        go_to("home")


# ---------------- REGISTER PAGE ---------------
def register_page():
    set_background("download (6).jpeg")

    st.markdown("""
        <style>

        /* --- Title (same as login) --- */
        .register-title {
            font-size: 48px;
            text-align: center;
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
            margin-bottom: 40px;
        }

        /* --- Input spacing (same as login) --- */
        .stTextInput {
            margin-bottom: 30px !important;
        }

        /* --- Center buttons and match width to inputs --- */
        div.stButton {
            text-align: center !important;
            width: 100% !important;
        }

        div.stButton > button {
            width: 100% !important;
            max-width: 450px !important; 
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            background: linear-gradient(135deg, #FF7E5F 0%, #FEB47B 30%) !important;
            border-radius: 20px !important;
            padding: 12px 40px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            color: white !important;
            cursor: pointer !important;
            border: none !important;
        }

        /* --- Input Field Styling (same as login) --- */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.2) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
            font-size: 16px !important;
            padding: 12px 20px !important;
        }

        .stTextInput > div > div > input:focus {
            border: 2px solid #FEB47B !important;
            box-shadow: 0 0 15px rgba(254, 180, 123, 0.4) !important;
        }

        /* Force BOTH buttons to have identical width */
        div.stButton > button {
            width: 100% !important;
            max-width: 200px !important;   /* SAME SIZE FOR BOTH BUTTONS */
            min-width: 200px !important;   /* LOCK WIDTH */
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
        }
        


        </style>
    """, unsafe_allow_html=True)

    # --- CENTER EVERYTHING (same structure as login) ---
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="register-title">Register Account</div>', unsafe_allow_html=True)

        username = st.text_input(
            "Registration Number / Employee ID",
            placeholder="Enter your ID",
            key="reg_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="reg_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            key="reg_confirm_password"
        )

        message_placeholder = st.empty()

        register_clicked = st.button("Register")
        back_clicked = st.button("Back to Home")

    # ------------- LOGIC -----------------
    if register_clicked:
        if not username or not password or not confirm_password:
            message_placeholder.error("Please fill in all fields.")
        elif password != confirm_password:
            message_placeholder.error("Passwords do not match.")
        else:
            # Determine role
            if re.match(r"^R\d{6}[A-Z]$", username):
                role = "student"
            elif re.match(r"^C\d{3}$", username):
                role = "counselor"
            elif re.match(r"^A\d{3}$", username):
                role = "admin"
            else:
                message_placeholder.warning("Unknown ID format. Please contact admin.")
                return

            success, msg = register_user(username, password, role)

            if success:
                message_placeholder.success(
                    f"{msg} You have been registered as {role.title()}. Redirecting..."
                )
                time.sleep(2)
                go_to("login")
            else:
                message_placeholder.error(msg)

    if back_clicked:
        go_to("home")




# ---------------- STUDENT PAGE ----------------
# def student_page():
#     # --- Set background ---
#     set_background("G2.jpeg")

#     # --- Load model and scaler ---
#     model = joblib.load("best_placement_model.pkl")
#     scaler = joblib.load("scaler.pkl")

#     # --- Page title ---
#     st.markdown(
#         "<h1 style='text-align:center; color:white; text-shadow:2px 2px 6px black; font-size:45px;'>"
#         "Student Job Placement Prediction</h1>",
#         unsafe_allow_html=True
#     )

#     # --- Registration number ---
#     reg_id = st.session_state.get("reg_number", "")
#     if not reg_id:
#         st.warning("Registration number not found. Please log in first.")
#         return
#     st.text_input("Registration Number", value=reg_id, disabled=True)

#     st.markdown("### Enter Your Details", unsafe_allow_html=True)

#     # ------------------------------
#     # BASIC INPUTS
#     # ------------------------------
#     col1, col2 = st.columns(2)

#     with col1:
#         cgpa = st.text_input("CGPA", placeholder="Enter CGPA (0.0 – 10.0)")
#     with col2:
#         comm = st.text_input("Communication Rating", placeholder="Rate 1–5")

#     # ------------------------------
#     # MORE INPUTS
#     # ------------------------------
#     with st.expander("More Inputs"):
#         internship = st.selectbox("Internship", ["Select...", "Yes", "No"])
#         hackathon = st.selectbox("Hackathon", ["Select...", "Yes", "No"])

#         backlogs = st.text_input("Backlogs", placeholder="Enter number of backlogs")
#         certs = st.text_input("Certifications", placeholder="Number of certificates")
#         major_proj = st.text_input("Major Projects", placeholder="Number of major projects")
#         mini_proj = st.text_input("Mini Projects", placeholder="Number of mini projects")
#         skills = st.text_input("Skill Rating", placeholder="Rate 1–10")

#     # ------------------------------
#     # VALIDATION CHECK
#     # ------------------------------
#     if st.button("Predict Placement"):

#         missing = []

#         if cgpa == "": missing.append("CGPA")
#         if comm == "": missing.append("Communication Rating")
#         if internship == "Select...": missing.append("Internship")
#         if hackathon == "Select...": missing.append("Hackathon")
#         if backlogs == "": missing.append("Backlogs")
#         if certs == "": missing.append("Certifications")
#         if major_proj == "": missing.append("Major Projects")
#         if mini_proj == "": missing.append("Mini Projects")
#         if skills == "": missing.append("Skill Rating")

#         if missing:
#             st.error(f"⚠️ Please fill the following fields before predicting: {', '.join(missing)}")
#             return

#         # ------------------------------
#         # SAFE NUMERIC CONVERSION
#         # ------------------------------
#         try:
#             CGPA = float(cgpa)
#             Comm = float(comm)
#             backlogs_val = int(backlogs)
#             Workshops = int(certs)
#             Major_Projects = int(major_proj)
#             Mini_Projects = int(mini_proj)
#             Skills = int(skills)

#             Internship = 1 if internship == "Yes" else 0
#             Hackathon = 1 if hackathon == "Yes" else 0

#         except:
#             st.error("Invalid input. Please check numeric fields.")
#             st.stop()

#         # ------------------------------
#         # FIXED VALUES USED DURING TRAINING
#         # ------------------------------
#         Pct12 = 12
#         Pct10 = 10

#         # ==============================
#         # FINAL 15 TRAINING FEATURES
#         # (MATCH TRAINING EXACTLY)
#         # ==============================
#         CGPA_x_10th = CGPA * Pct10
#         CGPA_x_12th = CGPA * Pct12
#         x10_x12 = Pct10 * Pct12
#         Academic_Avg = (CGPA + Pct10 + Pct12) / 3

#         # --- FINAL VECTOR (ORDER IS CRITICAL) ---
#         X = np.array([[ 
#             CGPA,
#             Major_Projects,
#             Workshops,
#             Mini_Projects,
#             Skills,
#             Comm,
#             Internship,
#             Hackathon,
#             Pct12,
#             Pct10,
#             backlogs_val,
#             CGPA_x_10th,
#             CGPA_x_12th,
#             x10_x12,
#             Academic_Avg
#         ]])

#         # ------------------------------
#         # SCALE + PREDICT
#         # ------------------------------
#         X_scaled = scaler.transform(X)
#         pred = model.predict(X_scaled)[0]
#         prob = model.predict_proba(X_scaled)[0][1]

#         # ------------------------------
#         # RESULT & RECOMMENDATION
#         # ------------------------------
#         if pred == 1:
#             result_text = " Likely to be Placed"
#             recommendation = "Strong predicted outcome! Keep on working hard to increase CGPA score, communication skills, " \
#             "overall skills and hackathon. Also aim to keep the number of your backlogs as low as possible."
#             color = "#00FF99"
#         else:
#             result_text = " At Risk"
#             recommendation = (
#                 "Your profile shows areas needing improvement. "
#                 "Increase study hours to improve overall CGPA and your skillset. Also keep your backlogs as low as possible"
#                 "while increase the number of internships and major projects."
#             )
#             color = "#FF6666"

#         # ------------------------------
#         # BUILD FEATURES FOR MONITORING
#         # ------------------------------
#         features_dict = {
#             "CGPA": CGPA,
#             "Communication": Comm,
#             "Internship": Internship,
#             "Hackathon": Hackathon,
#             "Backlogs": backlogs_val,
#             "Certifications": Workshops,
#             "Major_Projects": Major_Projects,
#             "Mini_Projects": Mini_Projects,
#             "Skills": Skills
#         }

#         # ------------------------------
#         # SAVE PREDICTION
#         # ------------------------------
#         save_prediction(
#             reg_id,
#             int(pred),
#             float(prob),
#             result_text,
#             recommendation,
#             features_dict
#         )

#         # ------------------------------
#         # DISPLAY RESULT CARD
#         # ------------------------------
#         st.html(f"""
#         <div style="
#             background-color: rgba(0,0,0,0.65);
#             padding: 25px;
#             border-radius: 15px;
#             box-shadow: 0 4px 15px rgba(0,0,0,0.4);
#             margin-top: 15px;
#             font-size:30px;">

#             <h2 style="color:white; text-align:center;">Prediction Result</h2>

#             <h3 style="color:{color}; text-align:center; font-size:30px;">
#                 {result_text}
#             </h3>

#             <p style="color:white; text-align:center; font-size:25px;">
#                 <strong>Placement Probability:</strong> {prob:.2%}
#             </p>

#             <p style="color:#cccccc; font-size:22px;">
#                 <strong>Recommendation:</strong> {recommendation}
#             </p>

#         </div>
#         """)

#     # ------------------------------
#     # BACK BUTTON
#     # ------------------------------
#     if st.button("Back to Home"):
#         for key in ["reg_number", "logged_in_user", "role"]:
#             st.session_state.pop(key, None)
#         go_to("home")


def student_page():
    # --- Set background ---
    set_background("G2.jpeg")

    # --- Load model and scaler ---
    model = joblib.load("best_placement_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # --- Page title ---
    st.markdown(
        "<h1 style='text-align:center; color:white; text-shadow:2px 2px 6px black; font-size:45px;'>"
        "Student Job Placement Prediction</h1>",
        unsafe_allow_html=True
    )

    # --- Registration number ---
    reg_id = st.session_state.get("reg_number", "")
    if not reg_id:
        st.warning("Registration number not found. Please log in first.")
        return
    st.text_input("Registration Number", value=reg_id, disabled=True)

    st.markdown("### Enter Your Details", unsafe_allow_html=True)

    # ------------------------------
    # BASIC INPUTS
    # ------------------------------
    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.text_input("CGPA", placeholder="Enter CGPA (0.0 – 10.0)")
    with col2:
        comm = st.text_input("Communication Rating", placeholder="Rate 1–5")

    # ------------------------------
    # MORE INPUTS
    # ------------------------------
    with st.expander("More Inputs"):
        internship = st.selectbox("Internship", ["Select...", "Yes", "No"])
        hackathon = st.selectbox("Hackathon", ["Select...", "Yes", "No"])
        backlogs = st.text_input("Backlogs", placeholder="Enter number of backlogs")
        certs = st.text_input("Certifications", placeholder="Number of certificates")
        major_proj = st.text_input("Major Projects", placeholder="Number of major projects")
        mini_proj = st.text_input("Mini Projects", placeholder="Number of mini projects")
        skills = st.text_input("Skill Rating", placeholder="Rate 1–10")

    # ------------------------------
    # PREDICTION BUTTON
    # ------------------------------
    if st.button("Predict Placement"):

        # ------------------------------
        # EMPTY FIELD VALIDATION
        # ------------------------------
        missing = []
        if cgpa == "": missing.append("CGPA")
        if comm == "": missing.append("Communication Rating")
        if internship == "Select...": missing.append("Internship")
        if hackathon == "Select...": missing.append("Hackathon")
        if backlogs == "": missing.append("Backlogs")
        if certs == "": missing.append("Certifications")
        if major_proj == "": missing.append("Major Projects")
        if mini_proj == "": missing.append("Mini Projects")
        if skills == "": missing.append("Skill Rating")

        if missing:
            st.error(f"⚠️ Please fill the following fields: {', '.join(missing)}")
            return

        # ------------------------------
        # SAFE NUMERIC CONVERSION
        # ------------------------------
        try:
            CGPA = float(cgpa)
            Comm = float(comm)
            backlogs_val = int(backlogs)
            Workshops = int(certs)
            Major_Projects = int(major_proj)
            Mini_Projects = int(mini_proj)
            Skills = int(skills)
            Internship = 1 if internship == "Yes" else 0
            Hackathon = 1 if hackathon == "Yes" else 0
        except Exception:
            st.error("❌ Invalid numeric input. Please check all numeric fields.")
            return

        # ------------------------------
        # FIXED VALUES (MATCH TRAINING)
        # ------------------------------
        Pct12 = 12
        Pct10 = 10

        CGPA_x_10th = CGPA * Pct10
        CGPA_x_12th = CGPA * Pct12
        x10_x12 = Pct10 * Pct12
        Academic_Avg = (CGPA + Pct10 + Pct12) / 3

        # ------------------------------
        # FINAL FEATURE VECTOR (ORDER CRITICAL)
        # ------------------------------
        X = np.array([[ 
            CGPA,
            Major_Projects,
            Workshops,
            Mini_Projects,
            Skills,
            Comm,
            Internship,
            Hackathon,
            Pct12,
            Pct10,
            backlogs_val,
            CGPA_x_10th,
            CGPA_x_12th,
            x10_x12,
            Academic_Avg
        ]], dtype=float)

        # ------------------------------
        # FINAL SAFETY VALIDATION
        # ------------------------------
        if np.isnan(X).any() or np.isinf(X).any():
            st.error("❌ Invalid input detected (NaN or Inf values).")
            return

        if X.shape != (1, 15):
            st.error(f"❌ Feature shape mismatch: expected (1, 15), got {X.shape}.")
            return

        # ------------------------------
        # SCALING + PREDICTION (SAFE)
        # ------------------------------
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"❌ Scaling failed: {e}")
            st.write("Debug values:", X)
            return

        if X_scaled.shape != (1, 15):
            st.error(f"❌ Scaled feature shape mismatch: {X_scaled.shape}")
            return

        try:
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
            st.write("Scaled input:", X_scaled)
            return

        # ------------------------------
        # RESULTS
        # ------------------------------
        if pred == 1:
            result_text = " Likely to be Placed"
            recommendation = (
                "Strong predicted outcome! Continue to improve CGPA, communication, "
                "skills, and project involvement. Keep backlogs low."
            )
            color = "#00FF99"
        else:
            result_text = " At Risk"
            recommendation = (
                "Improve CGPA, communication, skills, internships, and increase major "
                "projects while reducing backlogs."
            )
            color = "#FF6666"

        # ------------------------------
        # SAVE TO DATABASE
        # ------------------------------
        features_dict = {
            "CGPA": CGPA,
            "Communication": Comm,
            "Internship": Internship,
            "Hackathon": Hackathon,
            "Backlogs": backlogs_val,
            "Certifications": Workshops,
            "Major_Projects": Major_Projects,
            "Mini_Projects": Mini_Projects,
            "Skills": Skills
        }

        save_prediction(
            reg_id,
            int(pred),
            float(prob),
            result_text,
            recommendation,
            features_dict
        )

        # ------------------------------
        # DISPLAY RESULT
        # ------------------------------
        st.html(f"""
        <div style="
            background-color: rgba(0,0,0,0.65);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            margin-top: 15px;
            font-size:30px;">

            <h2 style="color:white; text-align:center;">Prediction Result</h2>

            <h3 style="color:{color}; text-align:center; font-size:30px;">
                {result_text}
            </h3>

            <p style="color:white; text-align:center; font-size:25px;">
                <strong>Placement Probability:</strong> {prob:.2%}
            </p>

            <p style="color:#cccccc; font-size:22px;">
                <strong>Recommendation:</strong> {recommendation}
            </p>

        </div>
        """)

    # ------------------------------
    # BACK BUTTON
    # ------------------------------
    if st.button("Back to Home"):
        for key in ["reg_number", "logged_in_user", "role"]:
            st.session_state.pop(key, None)
        go_to("home")





# ---------------- COUNSELOR PAGE ----------------
def counselor_page():
    set_background("download (6).jpeg")
    st.title("Counselor Dashboard")

    predictions = fetch_all_predictions()

    if not predictions:
        st.info("No student predictions yet.")
        if st.button("Back to Home"):
            for key in ["reg_number", "logged_in_user", "role"]:
                st.session_state.pop(key, None)
            go_to("home")
        return

    # Build DF
    df = pd.DataFrame(
        predictions,
        columns=[
            "reg_number",
            "pred_label",
            "pred_probability",
            "result_text",
            "recommendation",
            "features_json",
            "model_version",
            "created_at"
        ]
    )

    df["created_at"] = pd.to_datetime(df["created_at"])

    # Show overview table
    display_df = df[[
        "reg_number",
        "result_text",
        "pred_probability",
        "recommendation",
        "created_at"
    ]].copy()

    display_df.rename(columns={
        "reg_number": "Registration Number",
        "result_text": "Result",
        "pred_probability": "Probability",
        "recommendation": "Recommendation",
        "created_at": "Date"
    }, inplace=True)

    st.subheader("Student Predictions Overview")
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Explain an Individual Prediction")

    # Let counselor pick a specific row to explain
    # (we use the most recent prediction per student)
    # You could also use an index instead of reg_number if multiple.
    unique_ids = df["reg_number"].unique().tolist()
    selected_reg = st.selectbox("Select a student (Registration Number):", unique_ids)

    # Pick latest prediction for this student
    student_row = (
        df[df["reg_number"] == selected_reg]
        .sort_values("created_at", ascending=False)
        .iloc[0]
    )

    st.write(f"**Latest prediction for {selected_reg}:**")
    st.write(f"- Result: {student_row['result_text']}")
    st.write(f"- Probability: {student_row['pred_probability']:.2%}")
    st.write(f"- Recommendation: {student_row['recommendation']}")
    st.write(f"- Date: {student_row['created_at']}")

    if st.button("Explain This Prediction"):
        # store selection in session_state and go to explanation page
        st.session_state["explain_reg_number"] = selected_reg
        st.session_state["explain_created_at"] = str(student_row["created_at"])
        go_to("counselor_explain")

    if st.button("Back to Home"):
        for key in ["reg_number", "logged_in_user", "role"]:
            st.session_state.pop(key, None)
        go_to("home")



# ---------------- COUNSELOR EXPLANATION PAGE ----------------
def counselor_explain_page():
    set_background("download (6).jpeg")
    st.title("Student Prediction Explanation")

    # Guard: check selection
    if "explain_reg_number" not in st.session_state:
        st.warning("No student selected. Please go back to the Counselor Dashboard.")
        if st.button("Back to Counselor Dashboard"):
            go_to("counselor")
        return

    target_reg = st.session_state["explain_reg_number"]
    target_created_at = st.session_state.get("explain_created_at", None)

    st.write(f"### Explaining prediction for **{target_reg}**")

    # Reload predictions so we can get the full DF
    predictions = fetch_all_predictions()
    if not predictions:
        st.error("No predictions found in the database.")
        if st.button("Back to Counselor Dashboard"):
            go_to("counselor")
        return

    df = pd.DataFrame(
        predictions,
        columns=[
            "reg_number",
            "pred_label",
            "pred_probability",
            "result_text",
            "recommendation",
            "features_json",
            "model_version",
            "created_at"
        ]
    )
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Locate the exact row (reg_number + latest timestamp if needed)
    student_rows = df[df["reg_number"] == target_reg].copy()
    if student_rows.empty:
        st.error("Selected student's prediction could not be found.")
        if st.button("Back to Counselor Dashboard"):
            go_to("counselor")
        return

    # If we stored created_at, try to match that row, otherwise take latest
    if target_created_at is not None:
        try:
            ts = pd.to_datetime(target_created_at)
            student_row = student_rows.loc[student_rows["created_at"] == ts]
            if not student_row.empty:
                student_row = student_row.iloc[0]
            else:
                student_row = student_rows.sort_values("created_at", ascending=False).iloc[0]
        except Exception:
            student_row = student_rows.sort_values("created_at", ascending=False).iloc[0]
    else:
        student_row = student_rows.sort_values("created_at", ascending=False).iloc[0]

    # Display basic info
    st.markdown("#### Prediction Summary")
    st.write(f"- Result: **{student_row['result_text']}**")
    st.write(f"- Probability: **{student_row['pred_probability']:.2%}**")
    st.write(f"- Recommendation: {student_row['recommendation']}")
    st.write(f"- Date: {student_row['created_at']}")

    # ------------------------------------------------------
    # Reconstruct this student's raw & engineered features
    # ------------------------------------------------------
    feat = json.loads(student_row["features_json"])

    raw_dict = {
        "CGPA": float(feat["CGPA"]),
        "Major_Projects": int(feat["Major_Projects"]),
        "Workshops_Certificatios": int(feat["Certifications"]),
        "Mini_Projects": int(feat["Mini_Projects"]),
        "Skills": int(feat["Skills"]),
        "Communication_Skill_Rating": float(feat["Communication"]),
        "Internship": int(feat["Internship"]),
        "Hackathon": int(feat["Hackathon"]),
        "12th_Percentage": 12.0,
        "10th_Percentage": 10.0,
        "backlogs": int(feat["Backlogs"]),
    }

    X_raw_student = pd.DataFrame([raw_dict])
    X_eng_student = add_engineered_features(X_raw_student)
    feature_names = X_eng_student.columns.tolist()

    # Load model and scaler (same as in student_page)
    model = joblib.load("best_placement_model.pkl")
    scaler = joblib.load("scaler.pkl")

    X_scaled_student = scaler.transform(X_eng_student.values)

    # ------------------------------------------------------
    # Build SHAP + LIME explainers (using ALL predictions)
    # ------------------------------------------------------
    shap_explainer, lime_explainer, X_eng_all, X_scaled_all, feature_names_all = \
        get_explainers_and_background(model, scaler, df)

    if shap_explainer is None or lime_explainer is None:
        st.warning("Not enough data to compute SHAP/LIME explanations yet.")
        if st.button("Back to Counselor Dashboard"):
            go_to("counselor")
        return

    # ------------------------------------------------------
    # SHAP: local explanation
    # ------------------------------------------------------
    st.markdown("### SHAP Explanation (Local – This Student)")

    shap_vals = shap_explainer.shap_values(X_scaled_student)

    # Handle list output for binary classification
    if isinstance(shap_vals, list):
        shap_vals_student = shap_vals[0] if len(shap_vals) == 1 else shap_vals[1]
    else:
        shap_vals_student = shap_vals

    # Create a bar plot of top positive/negative contributions
    import matplotlib.pyplot as plt

    # sort by absolute impact
    abs_vals = np.abs(shap_vals_student[0])
    idx_sorted = np.argsort(-abs_vals)
    top_k = min(10, len(feature_names))
    top_idx = idx_sorted[:top_k]

    top_features = [feature_names[i] for i in top_idx]
    top_shap = shap_vals_student[0][top_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if v > 0 else "red" for v in top_shap]
    ax.barh(top_features[::-1], top_shap[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP Value (Impact on Placement Probability)")
    ax.set_title("Top Feature Contributions – SHAP (This Student)")
    plt.tight_layout()

    st.pyplot(fig)

    # Short textual explanation
    st.markdown("**Interpretation:**")
    st.write(
        "Features with **green bars** pushed the prediction towards **'Placed'**, "
        "while **red bars** pushed it towards **'At Risk'**. "
        "The longer the bar, the stronger the influence."
    )

    # ------------------------------------------------------
    # LIME: local explanation
    # ------------------------------------------------------
    # st.markdown("### LIME Explanation (Local – This Student)")

    # exp = lime_explainer.explain_instance(
    #     X_scaled_student[0],
    #     model.predict_proba,
    #     num_features=10
    # )

    # fig2 = exp.as_pyplot_figure()
    # plt.tight_layout()
    # st.pyplot(fig2)

    # st.markdown("**Interpretation:**")
    # st.write(
    #     "LIME shows how small changes around this student's profile affect the model's "
    #     "prediction. Features with positive weights support the current class, while "
    #     "negative weights oppose it."
    # )

    # ------------------------------------------------------
    # Raw features table
    # ------------------------------------------------------
    st.markdown("### Raw Input Features for This Student")
    st.dataframe(X_raw_student.T.rename(columns={0: "Value"}))

    if st.button("Back to Counselor Dashboard"):
        go_to("counselor")



        

# ---------------- ADMIN PAGE ----------------
def admin_page():
    set_background("download (9).jpeg")

    st.markdown("<h1 style='color:white;'>Admin Dashboard</h1>", unsafe_allow_html=True)

    predictions = fetch_all_predictions()

    if not predictions:
        st.info("No student predictions yet.")
        if st.button("Back to Home"):
            for key in ["reg_number", "logged_in_user", "role"]:
                st.session_state.pop(key, None)
            go_to("home")
        return

    # ===========================
    # Load real data
    # ===========================
    df = pd.DataFrame(predictions, columns=[
        "Registration Number",
        "pred_label",
        "pred_probability",
        "result_text",
        "recommendation",
        "features_json",
        "model_version",
        "created_at"
    ])
    df["created_at"] = pd.to_datetime(df["created_at"])



    # ==========================================================
    # SECTION 1 — OVERALL SUMMARY
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader(" Overall Prediction Summary")

    placed_count = (df["pred_label"] == 1).sum()
    not_placed_count = (df["pred_label"] == 0).sum()

    labels = ["Placed", "At Risk"]
    sizes = [placed_count, not_placed_count]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7, 3))  # Smaller & clean
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 12}
    )
    ax.axis("equal")

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.metric("Total Predictions", len(df))
    st.markdown("</div>", unsafe_allow_html=True)



    # ==========================================================
    # SECTION 2 — MODEL MONITORING
    # ==========================================================
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader(" Drift & Performance Overview ")
    # st.write("These charts show real changes in model behavior and student input patterns.")


    # -----------------------------------------------------------
    # 1 — TREND OVER TIME
    # -----------------------------------------------------------
    st.markdown("### Prediction Trend Over Time")

    trend_df = df.copy()
    trend_df["Day"] = trend_df["created_at"].dt.date
    daily_trend = trend_df.groupby("Day")["pred_label"].mean() * 100

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.line_chart(daily_trend)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Percentage of 'Placed' predictions per day.")



    # -----------------------------------------------------------
    # 3 — CGPA MEAN OVER TIME (DATA DRIFT)
    # -----------------------------------------------------------
    st.markdown("###  Data Drift Check — CGPA Mean Over Time")

    import json
    df["features"] = df["features_json"].apply(json.loads)
    df["CGPA"] = df["features"].apply(lambda f: f["CGPA"])
    cgpa_mean = df.groupby(df["created_at"].dt.date)["CGPA"].mean()

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.line_chart(cgpa_mean)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Tracks CGPA shifts over time.")

    st.markdown("</div>", unsafe_allow_html=True)



    # ==========================================================
    # BACK BUTTON
    # ==========================================================
    if st.button("Back to Home"):
        for key in ["reg_number", "logged_in_user", "role"]:
            st.session_state.pop(key, None)
        go_to("home")



if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "student":
    student_page()
elif st.session_state.page == "counselor":
    counselor_page()
elif st.session_state.page == "counselor_explain":
    counselor_explain_page()
elif st.session_state.page == "admin":
    admin_page()

