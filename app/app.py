import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

prob = 0.0
model = joblib.load("../plots_and_model/Heart_Disease_Model.pkl")

#Application configuration:
st.set_page_config(
    page_title="CAD Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.heart.org',
        'Report a bug': None,
        'About': "### Coronary Artery Disease Risk Predictor\n\nDeveloped by Ryan Naidoo\nStellenbosch University Data Science"
    }
)

st.title("Coronary Artery Disease Predictor")
st.caption("**Author:** Ryan Naidoo | **Affiliation:** Third-year Data Scientist Student at Stellenbosch Univeristy.")
st.info("AUC Score: 0.921 | Beats ECG 60-70% sensitivity | Correctly predicts 90% of heart disease patients.")

st.markdown("""
<style>
    /* Clean white background with subtle red accents */
    .stApp {
        background-color: black;
    }
    
    /* Red headers */
    h1, h2, h3 {
        color: #d32f2f;
        border-bottom: 2px solid #ffcdd2;
        padding-bottom: 10px;
    }
    
    /* Red buttons */
    .stButton > button {
        background-color: #d32f2f;
        color: white;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #b71c1c;
        box-shadow: 0 2px 8px rgba(211, 47, 47, 0.3);
    }
    
    /* Red metric values */
    [data-testid="stMetricValue"] {
        color: #d32f2f;
        font-size: 2rem;
    }
    
    /* Red border for containers */
    div[data-testid="stHorizontalBlock"] > div {
        border: 1px solid #ffcdd2;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    
    /* Red select boxes */
    .stSelectbox > div > div {
        border: 2px solid #ffcdd2;
    }
    
    /* Red slider */
    .stSlider > div > div > div {
        background-color: #d32f2f;
    }
    
    /* Medical red accent for important text */
    .medical-red {
        color: #black;
        font-weight: bold;
        background-color: #ffebee;
        padding: 5px 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

#Educational Section: 
with st.expander("Learn About Coronary Artery Disease (CAD)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is CAD?
        Coronary Artery Disease (CAD) is when plaque builds up in the arteries 
        that supply blood to your heart. This can lead to:
        
        - **Angina** (chest pain)
        - **Heart attacks**
        - **Heart failure**
        
        It is the **#1 cause of death** worldwide, causing nearly 610 000 deaths per year.
        
        ### Risk Factors:
        - High blood pressure
        - High cholesterol  
        - Smoking
        - Diabetes
        - Family history
        - Age (men >45, women >55)
        """)
    
    with col2:
        st.markdown("""
        ### How CAD is Diagnosed:
        1. **Physical exam** and medical history
        2. **Blood tests** (cholesterol, glucose)
        3. **ECG/EKG** - measures heart's electrical activity
        4. **Stress tests** - heart function during exercise
        5. **Cardiac catheterization** - direct artery imaging
        
        ### Prevention:
        - Healthy a diet that is low in salt and low in fat
        - Regular exercise (150 mins/week)
        - Not smoking
        - Regular medical check-ups at the doctor
        - Manage stress levels
        """)
    
    st.markdown("---")
    st.markdown("""
    ⚕️ **Source:** American Heart Association | **Dataset** : [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease/data)
    """)


with st.expander("Explore Feature Relationships", expanded=False):
    st.markdown("### Select a Feature to View Its Relationship with CAD")

    selected_feature = st.selectbox(
        "Choose a clinical feature:",
        ["Thallium Stress Test", "Exercise Angina", "Calcified Vessels"],
        help="See how each individual feature correlates with heart disease"
    )

    if selected_feature == "Thallium Stress Test":
        st.image("../plots_and_model/Thallium_and_CAD.png", caption="Thallium Stress Test Results vs CAD Diagnosis")

    elif selected_feature == "Exercise Angina":
        st.image("../plots_and_model/angina_vs_CAD.png", caption="Exercise Angina Presence vs CAD Diagnosis")


    else: 
        st.image("../plots_and_model/calc_vs_CAD.png", caption="Number of Calcified Vessels vs CAD Diagnosis")
    

##Clinical parameter section and the medical analysis:
st.markdown("## Enter Clinical Parameters:")
st.caption("Plesae provide the following key results from cardiac tests to assess your risk for CAD:")

select_col1, select_col2, select_col3 = st.columns(3)

with select_col1:
    with st.container(border=True):
        st.markdown("### Thallium Stress Test:")
        
        thal = st.selectbox(
            "Select your Thallium Stress Test result:",
            [3, 6, 7],
            format_func=lambda x: {
                3: "Normal (3) - No defects",
                6: "Fixed Defect (6) - Old damage", 
                7: "Reversible Defect (7) - Current reduced flow"
            }.get(x, str(x))
        )

with select_col2:  # Fixed: Changed from 'with select_col2:' to 'with select_col2:'
    with st.container(border=True):
        st.markdown("### Exercise Angina")
        
        exang = st.selectbox(
            "Exercise Angina",
            [0, 1],
            format_func=lambda x: {
                0: "No (0) - No chest pain during exercise",
                1: "Yes (1) - Chest pain during exercise"
            }.get(x, str(x)),
            help="Chest pain or discomfort when heart works harder during exercise, such as cardio."
        )

with select_col3:
    with st.container(border=True):
        st.markdown("### Number of Major Cardiovascular Vessels Calcified")
        
        ca = st.selectbox(
            "Select number of major vessels with >50% blockage:",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "None (0) - No significant blockages",
                1: "One (1) - Single vessel calcification",
                2: "Two (2) - Double vessel calcification", 
                3: "Three (3) - Triple vessel calcification"
            }.get(x, str(x)),
            help="Number of major coronary arteries with visible calcium deposits on fluoroscopy"
        )
        
if st.button("Predict Risk"):
    
    data = pd.DataFrame({'Thallium': [thal], 'Exercise_angina': [exang], 'Number_of_vessels_fluro': [ca]})
    
    with st.spinner("Running thallium stress analysis..."):
        prob = model.predict(data)[0] 
        
    st.success("✅ Analysis complete!")

st.metric("Coronary Artery Disease Risk (%)", f"{prob:.1%}")

def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text': "CAD Risk Gauge"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff4b4b"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},
                {'range': [30, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

#Risk guage.
st.plotly_chart(create_risk_gauge(prob), use_container_width=True)


#Medical Interpretation:
with st.expander("View Detailed Medical Interpretation:", expanded=False):
    st.markdown("""
    ### What These Results Mean:
    
    **🔴 High Risk (>70%):**
    - Strong likelihood of coronary artery disease.
    - Urgent cardiac evaluation needed.
    - Contact your medical services and see a doctor.
    
    **🟡 Moderate Risk (30-70%):**
    - Intermediate probability of CAD.
    - Further diagnostic tests and scans are recommended.
    - Lifestyle modifications are to be advised.
    
    **🟢 Low Risk (<30%):**
    - Low probability of significant CAD.
    - Preventative measures are still to be continued.
    - Regular follow-up recommended.
    
    **Disclaimer:** This tool is for educational and screening purposes only. I used the dataset and the machine learning model to practice building a medical project.
    Always consult with a healthcare professional for diagnosis. 
    """)


st.markdown("---")

st.markdown("""
    ## **Summary:**
    
    All of these three features show strong, clinically significant relationships with coronary artery disease. 
    This machine learning model combines these relationships to make an accurate prediction as to whether the patient exhibits CAD or not. Data Science demonstrates to be a powerful tool to integrate into health science, when it comes to making medical diagnostics like these. A qualified practioner would still need to be present to examine patients and perform cardiovascular tests, but integration with such technology can save lives and reduce CAD mortality rates worldwide!
    """)

# Minimal white box version
st.markdown("""
<style>
    .white-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .white-box img {
        width: 40px;
        height: 40px;
    }
    .white-box p {
        color: black;
        margin: 8px 0 0 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Contact Information: Let us collaborate on Computer Science and Data Science projects together!")

cols = st.columns(5)

links = [
    ("GitHub", "https://github.com/Ryan-Naidoo-05/Heart-Disease-Predictor-", "https://cdn-icons-png.flaticon.com/512/25/25231.png"),
    ("LinkedIn", "www.linkedin.com/in/ryan-naidoo-262838382", "https://cdn-icons-png.flaticon.com/512/174/174857.png"),
    ("Instagram", "https://www.instagram.com/ryan.naidoo_05?igsh=bmI3dmlnMnJtMHk4", "https://cdn-icons-png.flaticon.com/512/2111/2111463.png"),
    ("Email", "mailto:ryanpiman@gmail.com", "https://cdn-icons-png.flaticon.com/512/732/732200.png"),
    ("WhatsApp", "https://wa.me/0643509199", "https://cdn-icons-png.flaticon.com/512/220/220236.png")
]

for i, (name, url, icon) in enumerate(links):
    with cols[i]:
        st.markdown(f"""
        <a href="{url}" target="_blank" style="text-decoration: none;">
            <div class="white-box">
                <img src="{icon}">
                <p>{name}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

