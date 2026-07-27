import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="TinyML Human Activity Recognition",
    page_icon="🤖",
    layout="wide"
)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("activity_model.keras")

model = load_model()

activities = [
    "🚶 Walking",
    "⬆️ Walking Upstairs",
    "⬇️ Walking Downstairs",
    "🪑 Sitting",
    "🧍 Standing",
    "🛏️ Laying"
]

# -------------------- Custom CSS --------------------
st.markdown("""
<style>

.main {
    background-color:#f8fafc;
}

.big-title{
    font-size:48px;
    font-weight:800;
    color:#1565C0;
}

.subtitle{
    font-size:22px;
    color:#555555;
}

.card{
    background:#ffffff;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.12);
}

</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/artificial-intelligence.png",
    width=80
)

st.sidebar.title("TinyML HAR")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("---")

st.sidebar.subheader("Project Details")

st.sidebar.write("📌 Human Activity Recognition")

st.sidebar.write("📊 Dataset")
st.sidebar.info("UCI HAR Dataset")

st.sidebar.write("🧠 Framework")
st.sidebar.info("TensorFlow")

st.sidebar.write("⚡ Deployment")
st.sidebar.info("TinyML + Streamlit")

st.sidebar.write("🏷 Activities")
st.sidebar.success("6 Activities")

st.sidebar.markdown("---")

st.sidebar.write("Developed using")
st.sidebar.write("Python • TensorFlow • Streamlit")

# -------------------- Header --------------------

st.markdown(
    '<p class="big-title">🤖 TinyML Human Activity Recognition</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI Powered Human Activity Classifier using Deep Learning</p>',
    unsafe_allow_html=True
)

st.divider()

# -------------------- About --------------------

st.markdown("## 📖 About the Project")

st.markdown("""
This project uses a **Deep Learning Neural Network**
trained on the **UCI Human Activity Recognition Dataset**
to classify human activities using smartphone sensor data.

### Activities Recognized

- 🚶 Walking
- ⬆️ Walking Upstairs
- ⬇️ Walking Downstairs
- 🪑 Sitting
- 🧍 Standing
- 🛏️ Laying

The trained model has been converted to **TensorFlow Lite**
for TinyML deployment on edge devices.
""")

st.divider()
# ===============================
# AI Prediction Section
# ===============================

st.markdown("## 🤖 AI Prediction")

uploaded_file = st.file_uploader(
    "📂 Upload CSV File (One Sample with 561 Features)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read uploaded CSV
        data = pd.read_csv(uploaded_file, header=None).astype("float32")

        st.success("✅ File Uploaded Successfully")

        st.markdown("### Uploaded Sample")

        st.dataframe(data, use_container_width=True)

        if st.button("🔍 Predict Activity", use_container_width=True):

            with st.spinner("Analyzing sensor data..."):

                prediction = model.predict(data, verbose=0)

            predicted_class = np.argmax(prediction, axis=1)[0]

            confidence = float(prediction[0][predicted_class]) * 100

            st.markdown("---")

            st.markdown("# 🧠 Prediction Result")

            col1, col2 = st.columns([2,1])

            with col1:

                st.success(
                    f"### {activities[predicted_class]}"
                )

                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2f}%"
                )

                st.progress(confidence/100)

            with col2:

                st.info("Prediction completed successfully.")

            st.markdown("---")

            st.markdown("## 📊 Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Activity": activities,
                "Probability (%)": prediction[0]*100
            })

            st.bar_chart(
                probability_df.set_index("Activity")
            )

    except Exception as e:

        st.error(f"Error: {e}")
    # ===============================
# Model Performance
# ===============================

st.markdown("---")

st.markdown("# 📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Training Accuracy")
    st.image("accuracy.png", use_container_width=True)

with col2:
    st.subheader("📉 Training Loss")
    st.image("loss.png", use_container_width=True)

st.subheader("📊 Confusion Matrix")
st.image("confusion_matrix.png", use_container_width=True)

st.markdown("---")

# ===============================
# Model Information
# ===============================

st.markdown("# 📦 Model Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Activities", "6")

with c2:
    st.metric("Dataset", "UCI HAR")

with c3:
    st.metric("Framework", "TensorFlow")

with c4:
    st.metric("Deployment", "TinyML")

st.markdown("---")

# ===============================
# About TinyML
# ===============================

st.markdown("# ℹ️ About TinyML")

st.info("""
TinyML enables Machine Learning models to run on low-power embedded devices.

This project demonstrates how a Deep Learning model can be converted into
a lightweight TensorFlow Lite model suitable for deployment on edge devices
such as Arduino Nano 33 BLE Sense, ESP32, Raspberry Pi Pico, and similar
microcontrollers.
""")

st.markdown("---")

# ===============================
# Footer
# ===============================

st.markdown(
"""
<div style='text-align:center;padding:20px;'>

<h3>🤖 TinyML Human Activity Recognition</h3>

Built using ❤️ with

<b>Python • TensorFlow • Streamlit • TinyML</b>

<br><br>

Developed as an AI Project

</div>
""",
unsafe_allow_html=True
)