import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from google import genai
from dotenv import load_dotenv
import os
import time
from google.genai.errors import ServerError

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-flash-latest"


# ===================== SESSION STATE INIT ===================== #
if "proba" not in st.session_state:
    st.session_state["proba"] = None

if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False

if "user_data" not in st.session_state:
    st.session_state["user_data"] = None

if "ai_report" not in st.session_state:
    st.session_state["ai_report"] = None


# ===================== CONFIG ===================== #
st.set_page_config(
    page_title="Cardio Risk Prediction System",
    page_icon="🫀",
    layout="wide"
)

# ===================== THEMES ===================== #
st.markdown(f"""
<style>
 .simple-card {{
    background: #111827;           /* dark clean background */
    border: 1px solid #1f2937;     /* subtle border */
    border-radius: 12px;
    padding: 22px 18px;
    text-align: center;
}}

.simple-title {{
    font-size: 18px;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 8px;
}}
            
.simple-value {{
    font-size: 26px;
    font-weight: 700;
    color: #00d4ff;                /* cyan accent */
}}

.simple-sub {{
    font-size: 14px;
    color: #9ca3af;
    margin-top: 6px;
}}
/* Fade + slide animation */
.reveal {{
    opacity: 0;
    transform: translateY(40px);
    animation: revealAnim 1s ease forwards;
}}

.reveal-delay-1 {{ animation-delay: 0.2s;}}
.reveal-delay-2 {{ animation-delay: 0.4s;}}
.reveal-delay-3 {{ animation-delay: 0.6s; }}

@keyframes revealAnim {{
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

/* Image styling */
.hero-img {{
    width: 100%;
    border-radius: 18px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
}}

/* Section title */
.hero-title {{
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 15px;
}}

/* Paragraph */
.hero-text {{
    font-size: 17px;
    line-height: 1.8;
    color: #d1d5db;
}}
/* Neon Training Card */
.neon-card {{
    position: relative;
    background: linear-gradient(145deg, #0b0f14, #111827);
    border-radius: 18px;
    padding: 28px 20px;
    height: 100%;
    color: #ffffff;
    text-align: center;

    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.6);
    transition: all 0.4s ease-in-out;
    overflow: hidden;
}}

/* Left glow line */
.neon-card::before {{
    content: "";
    position: absolute;
    top: 15%;
    left: 0;
    width: 3px;
    height: 70%;
    background: linear-gradient(to bottom, transparent, #00d4ff, transparent);
}}

/* Right glow line */
.neon-card::after {{
    content: "";
    position: absolute;
    top: 15%;
    right: 0;
    width: 3px;
    height: 70%;
    background: linear-gradient(to bottom, transparent, #00d4ff, transparent);
}}

/* Hover effect */
.neon-card:hover {{
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 20px 55px rgba(0, 212, 255, 0.35);
}}

/* Title */
.neon-title {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 12px;
}}

/* Value */
.neon-value {{
    font-size: 28px;
    font-weight: 800;
    color: #00d4ff;
}}

/* Sub text */
.neon-sub {{
    margin-top: 8px;
    font-size: 14px;
    color: #b5c7d3;
}}

.report-card {{
    background: linear-gradient(145deg, #0b0f14, #111827);
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
    color: #e5e7eb;
    box-shadow: 0 20px 45px rgba(0, 212, 255, 0.25);
    border-left: 5px solid #00d4ff;
}}
            
.report-title {{
    font-size: 24px;
    font-weight: 800;
    color: #00d4ff;
    margin-bottom: 12px;
}}

.report-text {{
    font-size: 16px;
    line-height: 1.8;
    white-space: pre-wrap;
}}

</style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL ===================== #
@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ===================== SIDEBAR ===================== #
st.sidebar.markdown("## 📂 Navigation")
def sidebar_button(label, page_name, icon=""):
    is_active = st.session_state.get("page", "Home") == page_name

    if is_active:
        st.sidebar.markdown(
            f"""
            <div style="
                padding: 10px;
                margin-bottom: 6px;
                border-radius: 8px;
                background-color: rgba(0, 212, 255, 0.15);
                border-left: 5px solid #00d4ff;
                font-weight: 700;
            ">
                {icon} {label}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(f"{icon} {label}", use_container_width=True):
            st.session_state["page"] = page_name
            st.rerun()



sidebar_button("Home", "Home", "🏠")
sidebar_button("About Cardiology", "About", "🫀")
sidebar_button("Test Model", "Test", "🧪")
sidebar_button("Model Performance", "Performance", "📊")

page = st.session_state.get("page", "Home")


if page == "Home":

    # ---------- TOP SECTION ----------
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("""
        <div class="reveal reveal-delay-1">
            <div class="hero-title">🫀 Cardiology </div>
            <div class="hero-text">
                Cardiology is the branch of medicine that focuses on the diagnosis,
                treatment, and prevention of diseases related to the heart and blood vessels.
                With the help of modern data-driven techniques and artificial intelligence,
                cardiovascular risks can now be identified at an early stage.
            </div>
            <br>
            <div class="hero-text">
                This system combines medical knowledge with machine learning to assist
                in understanding heart health and predicting cardiovascular disease risk.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="reveal reveal-delay-2 hero-text">
            <ul>
                <li>💓 Monitors heart-related risk factors</li>
                <li>📊 Uses clinical and lifestyle parameters</li>
                <li>🧠 Applies machine learning for prediction</li>
                <li>⚕️ Supports preventive healthcare decisions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.image(
            "assets/images/cardiology_dashboard.png",
            use_container_width=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------- IMAGE GALLERY ----------
    st.markdown("""
    <div class="reveal reveal-delay-1 hero-title" style="font-size:32px;">
        ❤️ Understanding the Human Heart
    </div>
    """, unsafe_allow_html=True)

    img1, space1, img2= st.columns([1, 1, 1])


    with img1:
        st.image("assets/images/Heart_1.png", use_container_width=True)
    with img2:
        st.image("assets/images/Heart_2.png", use_container_width=True)
    # with img3:
    #     st.image("assets/images/Heart_3.png", use_container_width=True)
    # with img4:
    #     st.image("https://images.unsplash.com/photo-1530026405186-ed1f139313f8", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- IMPORTANCE SECTION ----------
    st.markdown("""
    <div class="reveal reveal-delay-2 hero-text">
        <b>Why Heart Health Is Important</b><br><br>
        The heart is one of the most vital organs in the human body. It continuously
        pumps blood to supply oxygen and nutrients to all organs. Poor heart health
        can lead to serious conditions such as heart attacks, strokes, and heart failure.
        <br><br>
        Maintaining a healthy heart involves regular physical activity, balanced nutrition,
        stress management, and timely medical check-ups. Early prediction systems like this
        help individuals become aware of their cardiovascular risk and take preventive action.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 🫀 ABOUT CARDIOLOGY
# =====================================================
elif page == "About":

    st.markdown("<div class='title'>🫀 Cardiovascular Health & AI Insights</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🫀 About Cardiology", "🤖 AI Prediction in Cardiology"])

    # =================================================
    # TAB 1: ABOUT CARDIOLOGY
    # =================================================
    with tab1:

        st.markdown("""
        <div class='card'>
        Cardiovascular disease (CVD) refers to a group of disorders affecting the heart
        and blood vessels. It remains one of the leading causes of death worldwide.
        Early identification of risk factors plays a crucial role in prevention and
        long-term heart health management.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📌 Key Risk Factors")
        st.markdown("""
        - 🩸 High blood pressure  
        - ⚖️ Obesity and high Body Mass Index (BMI)  
        - 🚬 Smoking and alcohol consumption  
        - 🍬 Diabetes and high glucose levels  
        - 🧍 Lack of physical activity  
        """)

        # -------- DONUT CHART (Neon Colors) -------- #
        age_df = pd.DataFrame({
            "Age Group": ["18–35", "36–50", "51–65", "65+"],
            "CVD Risk (%)": [12, 28, 38, 22]
        })

        fig = px.pie(
            age_df,
            names="Age Group",
            values="CVD Risk (%)",
            hole=0.55,   # 🔥 DONUT
            color_discrete_sequence=[
                "#00d4ff", "#38bdf8", "#818cf8", "#22d3ee"
            ]
        )

        fig.update_layout(
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='card'>
        The donut chart highlights how cardiovascular risk increases with age.
        Older age groups are generally more vulnerable due to prolonged exposure
        to lifestyle and metabolic risk factors. However, unhealthy habits at a
        younger age can significantly accelerate heart disease development.
        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # TAB 2: AI PREDICTION IN CARDIOLOGY
    # =================================================
    with tab2:

        left, right = st.columns([1.3, 1])

        with left:
            st.markdown("""
            <div class='card'>
            <b>🤖 Role of Artificial Intelligence in Cardiology</b><br><br>
            Artificial Intelligence (AI) is increasingly used in cardiology to
            analyze large volumes of clinical and lifestyle data. AI models can
            detect complex patterns that may not be easily noticeable by humans.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='card'>
            <b>✅ Advantages of AI Prediction</b><br>
               • Fast and consistent decision-making<br>
               • Can analyze thousands of records simultaneously<br>
               • Reduces human bias<br>
               • Supports early detection and prevention
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='card'>
            <b>⚠️ Limitations of AI Prediction</b><br>
            • Depends heavily on data quality<br>
            • Lacks emotional and contextual understanding<br>
            • Cannot replace clinical expertise<br>
            • Requires validation by medical professionals
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.image(
                "assets/images/ai_cardiology.png",
                use_container_width=True
            )

        # -------- AI vs HUMAN COMPARISON GRAPH -------- #
        compare_df = pd.DataFrame({
            "Decision Maker": ["AI System", "Human Doctor"],
            "Accuracy (%)": [78, 85]
        })

        fig2 = px.bar(
            compare_df,
            x="Decision Maker",
            y="Accuracy (%)",
            color="Decision Maker",
            color_discrete_sequence=["#00d4ff", "#a88eea"]
        )

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class='card'>
        AI systems are designed to <b>assist</b> doctors, not replace them.
        While AI offers speed and data-driven insights, human doctors bring
        experience, ethical judgment, and patient-centered care.
        The best outcomes are achieved when AI and human expertise work together.
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 🧪 TEST MODEL
# =====================================================
elif page == "Test":

    st.markdown("<div class='title'>🧪 Heart Risk Assessment</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📘 Training Summary", "🔬 Live Prediction"])

    # ---------------- TRAINING INFO ---------------- #

    with tab1:
        col1, space1, col2, space2, col3 = st.columns([1, 0.15, 1, 0.15, 1])

        with col1:
            st.markdown("""
            <div class="simple-card">
                <div class="simple-title">📊 Total Records</div>
                <div class="simple-value">68,323</div>
                <div class="simple-sub">Cleaned training dataset</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="simple-card">
                <div class="simple-title">✅ Positive Cases</div>
                <div class="simple-value">34,100</div>
                <div class="simple-sub">Patients with CVD</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="simple-card">
                <div class="simple-title">❌ Negative Cases</div>
                <div class="simple-value">34,223</div>
                <div class="simple-sub">Healthy individuals</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        col4, space3, col5, space4, col6 = st.columns([1, 0.15, 1, 0.15, 1])

        with col4:
            st.markdown("""
            <div class="simple-card">
                <div class="simple-title">👶 Young Patients</div>
                <div class="simple-value">18–45</div>
                <div class="simple-sub">Lower age risk group</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown("""
            <div class="simple-card">
                <div class="simple-title">👴 Older Patients</div>
                <div class="simple-value">45+</div>
                <div class="simple-sub">Higher risk group</div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown("""
            <div class="simple-card">
                <div class="simple-title">🧹 Outliers Removed</div>
                <div class="simple-value">~200</div>
                <div class="simple-sub">Improved model stability</div>
            </div>
            """, unsafe_allow_html=True)



    # ---------------- LIVE TEST ---------------- #
    def generate_cardiology_report(prompt: str) -> str:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    
    def generate_cardiology_report(prompt):
        for attempt in range(3):  # try 3 times
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                return response.text
            except ServerError:
                time.sleep(2)  # wait 2 seconds and retry

        return "⚠️ AI service is currently busy. Please try again after some time."
    
    def build_prompt(user_data, proba):
        risk_percent = round(proba[1] * 100, 2)
        risk_label = "HIGH RISK" if proba[1] > 0.5 else "LOW RISK"

        return f"""
    You are a medical AI assistant.

    Patient Information:
    - Age: {user_data['age_year']} years
    - Gender: {user_data['gender']}
    - BMI: {user_data['BMI']}
    - Blood Pressure: {user_data['ap_hi']}/{user_data['ap_lo']}
    - Cholesterol: {user_data['cholesterol']}
    - Glucose: {user_data['gluc']}
    - Smoking: {user_data['smoke']}
    - Physical Activity: {user_data['active']}

    ML Model Output:
    - Cardiovascular Risk: {risk_label}
    - Probability: {risk_percent}%

    Write a professional cardiology report including:
    1. Risk interpretation
    2. Key contributing factors
    3. Lifestyle recommendations
    4. Medical disclaimer (educational only)
    """

    

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            age_year = st.number_input("Age (Years)", 18, 100, 40)
            height = st.number_input("Height (cm)", 140, 210, 165)
            weight = st.number_input("Weight (kg)", 40.0, 180.0, 70.0)
            gender = st.selectbox("Gender", ["Female", "Male"])

        with c2:
            ap_hi = st.number_input("Systolic BP", 90, 240, 120)
            ap_lo = st.number_input("Diastolic BP", 60, 140, 80)
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "High"])
            gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "High"])

        smoke = st.checkbox("Smoking")
        alco = st.checkbox("Alcohol")
        active = st.checkbox("Physically Active")

        BMI = round(weight / ((height / 100) ** 2), 2)
        pulse_pressure = ap_hi - ap_lo
        hypertension = 1 if ap_hi >= 140 else 0

        if st.button("🔮 Analyze Risk"):
            X_new = np.array([[
                1 if gender == "Female" else 2,
                height, weight, ap_hi, ap_lo,
                {"Normal":1,"Above Normal":2,"High":3}[cholesterol],
                {"Normal":1,"Above Normal":2,"High":3}[gluc],
                int(smoke), int(alco), int(active),
                age_year, BMI, pulse_pressure, hypertension
            ]])
            st.session_state["proba"] = model.predict_proba(scaler.transform(X_new))[0]
            st.session_state["prediction_done"] = True
            st.session_state["user_data"] = {
                "age_year": age_year,
                "gender": gender,
                "BMI": BMI,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "pulse_pressure": pulse_pressure,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": "Yes" if smoke else "No",
                "alco": "Yes" if alco else "No",
                "active": "Yes" if active else "No"
            }
            # proba = model.predict_proba(scaler.transform(X_new))[0]
        if st.session_state["prediction_done"]:
            proba = st.session_state["proba"]
            st.subheader("📊 Prediction Result")
            if proba[1] > 0.5:
                st.error(f"⚠️ HIGH RISK ({proba[1]*100:.2f}%)")
            else:
                st.success(f"✅ LOW RISK ({proba[1]*100:.2f}%)") 
            # st.success(risk)

            st.metric("Positive Risk Probability", f"{proba[1]*100:.2f}%")
            st.metric("Negative Risk Probability", f"{proba[0]*100:.2f}%")

            bmi_status = (
                "😟 Underweight" if BMI < 18.5 else
                "😊 Normal" if BMI < 25 else
                "😐 Overweight" if BMI < 30 else
                "⚠️ Obese"
            )

            st.markdown(f"""
            <div class='card'>
            📏 BMI: **{BMI}** → {bmi_status}  
            <br>
            💓 Pulse Pressure: **{pulse_pressure}**  
            <br>
            🩸 Hypertension: **{"Yes" if hypertension else "No"}**
            </div>
            """, unsafe_allow_html=True)

            chart = pd.DataFrame({
                "Type": ["No Disease", "Disease"],
                "Probability": proba
            })

            fig = px.bar(
                chart,
                x="Type",
                y="Probability",
                color="Type",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            st.plotly_chart(fig, use_container_width=True)

        if st.button("🧠 Generate AI Cardiology Report"):
            if not st.session_state["prediction_done"]:
                st.warning("⚠️ Please analyze risk first.")
            else:
                prompt = build_prompt(
                    st.session_state["user_data"],
                    st.session_state["proba"]
                )
                with st.spinner("🧠 Generating AI cardiology report..."):
                    report = generate_cardiology_report(prompt)
                    st.session_state["ai_report"] = report
                if st.session_state.get("ai_report"):
                    st.markdown(f"""
                    <div class="report-card">
                        <div class="report-title">🧠 AI Cardiology Report</div>
                        <div class="report-text">
                            {st.session_state["ai_report"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        if st.session_state.get("ai_report"):
            st.download_button(
                label="📥 Download Report (TXT)",
                data=st.session_state["ai_report"],
                file_name="AI_Cardiology_Report.txt",
                mime="text/plain"
            )


# =====================================================
# 📊 MODEL PERFORMANCE
# =====================================================
elif page == "Performance":
    st.markdown("<div class='title'>📊 Model Performance Overview</div>", unsafe_allow_html=True)

    # ---------- SUMMARY CARD ----------
    st.markdown("""
    <div class='card'>
    <b>Overall Model Results</b><br><br>
    ✔ Accuracy: <b>~78%</b><br>           
    ✔ Balanced Dataset (Stratified Sampling)<br>
    ✔ No Data Leakage<br> 
    ✔ Clinically Relevant Features Used<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 Model Development Pipeline")

    c1, c2, c3 = st.columns(3)

    # ---------- CARD 1: EDA ----------
    with c1:
        st.markdown("""
        <div class="neon-card">
            <div class="neon-title">🔍 Exploratory Data Analysis</div>
            <div class="neon-sub">
            • Analyzed age, BP, BMI distributions<br>
            • Identified missing & abnormal values<br>
            • Studied correlations with CVD<br>
            • Visualized risk trends
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- CARD 2: PREPROCESSING ----------
    with c2:
        st.markdown("""
        <div class="neon-card">
            <div class="neon-title">🧹 Data Preprocessing</div>
            <div class="neon-sub">
            • Removed medical outliers<br>
            • Feature scaling (StandardScaler)<br>
            • Encoded categorical features<br>
            • Balanced class distribution
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- CARD 3: FEATURE ENGINEERING ----------
    with c3:
        st.markdown("""
        <div class="neon-card">
            <div class="neon-title">⚙️ Feature Engineering</div>
            <div class="neon-sub">
            • BMI calculation<br>
            • Pulse pressure feature<br>
            • Hypertension flag<br>
            • Lifestyle indicators
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)

    # ---------- CARD 4: MODEL TRAINING ----------
    with c4:
        st.markdown("""
        <div class="neon-card">
            <div class="neon-title">🤖 Model Training</div>
            <div class="neon-sub">
            • Logistic Regression model<br>
            • Binary classification (CVD / No CVD)<br>
            • Stratified train-test split<br>
            • Optimized for interpretability
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- CARD 5: MODEL EVALUATION ----------
    with c5:
        st.markdown("""
        <div class="neon-card">
            <div class="neon-title">📊 Model Evaluation</div>
            <div class="neon-sub">
            • Accuracy ≈ 78%<br>
            • Probability-based prediction<br>
            • Stable generalization<br>
            • No overfitting detected
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- CARD 6: AI INTEGRATION ----------
    with c6:
        st.markdown("""
        <div class="neon-card">
            <div class="neon-title">🧠 AI Explainability</div>
            <div class="neon-sub">
            • Gemini AI integration<br>
            • Human-readable reports<br>
            • Risk interpretation<br>
            • Lifestyle recommendations
            </div>
        </div>
        """, unsafe_allow_html=True)


    # ---------- FEATURE ENGINEERING ----------
    st.markdown("### ⚙️ Feature Engineering Highlights")

    st.markdown("""
    <div class='card'>
    ✔ Body Mass Index (BMI) = Weight / Height²<br>
    ✔ Pulse Pressure = Systolic BP − Diastolic BP<br>
    ✔ Hypertension Indicator based on clinical threshold<br>
    ✔ Lifestyle indicators (Smoking, Alcohol, Physical Activity)<br><br>
    These engineered features improved model interpretability and prediction stability.
    </div>
    """, unsafe_allow_html=True)

    # ---------- SIMPLE PERFORMANCE CHART ----------
    perf_df = pd.DataFrame({
        "Metric": ["Accuracy", "Error Rate"],
        "Value": [78, 22]
    })

    fig = px.bar(
        perf_df,
        x="Metric",
        y="Value",
        color="Metric",
        color_discrete_sequence=["#00d4ff", "#f87171"],
        text="Value"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis_title="Percentage (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- AI EXPLAINABILITY ----------
    st.markdown("### 🧠 AI Explainability & Report Generation")

    st.markdown("""
    <div class='card'>
    To enhance model interpretability, an AI-based analysis module was integrated
    using <b>Google Gemini (Flash Model)</b>.

    <br><br>
    <b>Role of AI in This System:</b><br>
    • Interprets machine learning probability outputs<br>
    • Explains cardiovascular risk in human-readable language<br>
    • Highlights contributing health factors<br>
    • Provides lifestyle guidance for preventive care<br><br>

    The AI does <b>not</b> perform diagnosis. It acts as a supportive tool to help
    users understand model predictions more clearly.
    </div>
    """, unsafe_allow_html=True)

    # ---------- DISCLAIMER ----------
    st.info(
        "⚠️ This system is intended for **educational and screening purposes only**. "
        "It does not replace professional medical consultation or diagnosis."
    )
