import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page config
st.set_page_config(
    page_title="FireGuard AI",
    page_icon="🧯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, more intentional dashboard look
st.markdown("""
    <style>
    :root {
        --bg: #0b1020;
        --panel: rgba(14, 20, 39, 0.82);
        --panel-border: rgba(255, 255, 255, 0.08);
        --text: #edf2ff;
        --muted: #94a3b8;
        --accent: #ff5b5f;
        --accent-2: #ff9a5a;
        --success: #2dd4bf;
        --success-bg: rgba(45, 212, 191, 0.14);
        --danger-bg: rgba(248, 113, 113, 0.14);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 91, 95, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(255, 154, 90, 0.12), transparent 22%),
            linear-gradient(180deg, #08101f 0%, #0b1327 42%, #09101d 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    .hero {
        padding: 1.6rem 1.5rem 1.2rem;
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(13, 18, 34, 0.78));
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.2rem;
    }

    .main-header {
        font-size: 3.1rem;
        font-weight: 800;
        color: var(--text);
        text-align: center;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: var(--muted);
        text-align: center;
        margin-bottom: 0.75rem;
    }

    .kpi-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1rem 0 0.2rem;
    }

    .kpi {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid var(--panel-border);
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(8px);
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.1;
    }

    .kpi-delta {
        margin-top: 0.25rem;
        font-size: 0.82rem;
        color: #cbd5e1;
    }

    .panel {
        border: 1px solid var(--panel-border);
        border-radius: 22px;
        background: var(--panel);
        padding: 1.15rem;
        box-shadow: 0 18px 38px rgba(0, 0, 0, 0.26);
    }

    .panel h3 {
        color: var(--text);
        margin-top: 0;
    }

    .panel-label {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0.55rem;
    }

    .tag {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 0.8rem;
    }

    .tag-success {
        background: var(--success-bg);
        color: #bdf7ee;
        border: 1px solid rgba(45, 212, 191, 0.25);
    }

    .tag-danger {
        background: var(--danger-bg);
        color: #ffd0d0;
        border: 1px solid rgba(248, 113, 113, 0.25);
    }

    .tag-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #dbe7ff;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .compliant {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.18), rgba(45, 212, 191, 0.1));
        border: 1px solid rgba(45, 212, 191, 0.25);
        padding: 1rem 1.1rem;
        border-radius: 18px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        color: #d6fff7;
        margin-bottom: 0.85rem;
    }
    .non-compliant {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.18), rgba(255, 154, 90, 0.08));
        border: 1px solid rgba(248, 113, 113, 0.25);
        padding: 1rem 1.1rem;
        border-radius: 18px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffe0e0;
        margin-bottom: 0.85rem;
    }

    div[data-testid="stTabs"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        padding: 0.75rem 0.9rem 0.25rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 0.35rem;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 0.65rem 0.8rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 12, 24, 0.97), rgba(13, 18, 34, 0.98));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] .sidebar-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.9rem;
    }

    [data-testid="stSidebar"] .sidebar-card ul {
        margin: 0.35rem 0 0 1rem;
        padding: 0;
    }

    [data-testid="stSidebar"] .sidebar-card li {
        margin-bottom: 0.35rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text);
        margin: 0.2rem 0 0.7rem;
    }

    .subtle-text {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.85rem 0 1rem;
    }

    .dash-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 0.95rem 1rem;
    }

    .dash-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .dash-value {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .dash-note {
        color: #cbd5e1;
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }

    .confidence-box {
        margin: 0.3rem 0 0.85rem;
        padding: 0.85rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
        color: #dbe7ff;
    }

    .confidence-box strong {
        color: var(--accent-2);
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin: 0.35rem 0 0.85rem;
    }

    .mini-status {
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
        color: #dbe7ff;
        font-size: 0.86rem;
        font-weight: 700;
    }

    .mini-status strong {
        color: var(--accent-2);
    }

    .result-shell {
        margin-top: 0.7rem;
        padding: 1rem;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.84), rgba(12, 17, 31, 0.94));
        border: 1px solid var(--panel-border);
    }

    .image-frame {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.09);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

sidebar = st.sidebar
sidebar.markdown("## FireGuard AI")
sidebar.markdown("<div class='sidebar-card'><div class='subtle-text'>Upload a room or corridor image, set the confidence threshold, and check whether a fire extinguisher is visible.</div></div>", unsafe_allow_html=True)
sidebar.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
sidebar.markdown("**How to use**")
sidebar.markdown("- Upload a clear image\n- Adjust the confidence threshold\n- Read the compliance result\n- Inspect the annotated output")
sidebar.markdown("</div>", unsafe_allow_html=True)
sidebar.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
sidebar.markdown("**Threshold guide**")
sidebar.markdown("<div class='subtle-text'>Lower values detect more objects but can add false positives. Higher values are stricter and only keep stronger detections.</div>", unsafe_allow_html=True)
sidebar.markdown("</div>", unsafe_allow_html=True)
sidebar.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
sidebar.markdown("**Tips**")
sidebar.markdown("- Use well-lit images\n- Keep the extinguisher visible\n- Avoid heavy blur or occlusion")
sidebar.markdown("</div>", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="hero">
        <p class="main-header">🧯 FireGuard AI</p>
        <p class="sub-header">Automated Fire Extinguisher Compliance Detection System</p>
        <div class="kpi-row">
            <div class="kpi">
                <div class="kpi-label">Model</div>
                <div class="kpi-value">YOLOv8s</div>
                <div class="kpi-delta">Trained on custom compliance data</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Use case</div>
                <div class="kpi-value">Safety Audit</div>
                <div class="kpi-delta">Room and corridor inspection</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Status</div>
                <div class="kpi-value">Ready</div>
                <div class="kpi-delta">Upload an image to detect</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.05, 1.25], gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Control Panel</div>', unsafe_allow_html=True)
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Upload a room/corridor image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Lower values show more detections, higher values keep only stronger detections."
    )
    threshold_note = "Stricter" if confidence >= 0.7 else "Balanced" if confidence >= 0.4 else "Sensitive"
    st.markdown(
        f'<div class="confidence-box"><strong>{threshold_note}</strong> threshold selected at <strong>{confidence:.2f}</strong>. Use this to control how confident the model must be before it draws a box.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="status-row"><div class="mini-status">Status: <strong>{"Ready" if uploaded is None else "Analyzing"}</strong></div><div class="mini-status">Confidence: <strong>{confidence:.2f}</strong></div><div class="mini-status">Mode: <strong>{threshold_note}</strong></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Tip: use a clear room or corridor image for best detection results.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Monitoring Dashboard</div>', unsafe_allow_html=True)
    if uploaded is not None:
        image = Image.open(uploaded)
        results = model(image, conf=confidence)
        count = len(results[0].boxes)
        top_conf = float(results[0].boxes.conf.max()) if count > 0 else 0.0
        compliance = "COMPLIANT" if count > 0 else "NON-COMPLIANT"
        compliance_class = "compliant" if count > 0 else "non-compliant"

        st.markdown(
            f'<div class="{compliance_class}">{"✅" if count > 0 else "❌"} {compliance}<br>{count} Fire Extinguisher(s) Detected</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="dashboard-grid">
                <div class="dash-card">
                    <div class="dash-label">Detections</div>
                    <div class="dash-value">{count}</div>
                    <div class="dash-note">Objects above threshold</div>
                </div>
                <div class="dash-card">
                    <div class="dash-label">Confidence</div>
                    <div class="dash-value">{top_conf:.0%}</div>
                    <div class="dash-note">Highest detected score</div>
                </div>
                <div class="dash-card">
                    <div class="dash-label">Threshold</div>
                    <div class="dash-value">{confidence:.2f}</div>
                    <div class="dash-note">Current filter setting</div>
                </div>
            </div>
            """.format(count=count, top_conf=top_conf, confidence=confidence),
            unsafe_allow_html=True,
        )

        st.subheader("Detection Result")
        result_image = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_image_rgb, width="stretch")

        st.markdown('<div class="result-shell">', unsafe_allow_html=True)
        st.markdown("**Detection Details**")
        if count > 0:
            detection_rows = []
            for i, box in enumerate(results[0].boxes):
                conf_score = float(box.conf[0])
                st.write(f"🧯 Fire Extinguisher {i+1}: **{conf_score:.1%}** confidence")
                detection_rows.append({"Object": f"Fire Extinguisher {i + 1}", "Confidence": f"{conf_score:.1%}"})
            st.table(detection_rows)
        else:
            st.write("No fire extinguisher confidence scores to show because no detections were returned.")
            st.markdown('<div class="tag tag-neutral">No detection table to display</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="tag tag-danger">No image uploaded yet</div>',
            unsafe_allow_html=True
        )
        st.info("Upload a room or corridor image on the left to start detection.")
        st.image(
            "https://via.placeholder.com/900x520/101826/edf2ff?text=Upload+an+Image+to+Detect",
            width="stretch"
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.subheader("Model Performance")
perf1, perf2, perf3, perf4 = st.columns(4)
with perf1:
    st.metric(label="Precision", value="98%", delta="+18% vs target")
with perf2:
    st.metric(label="Recall", value="95%", delta="+17% vs target")
with perf3:
    st.metric(label="mAP@50", value="95%", delta="+13% vs target")
with perf4:
    st.metric(label="mAP@50-95", value="80%", delta="+15% vs target")

st.markdown("---")
st.subheader("Training Details")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | Model | YOLOv8s |
    | Epochs | 50 |
    | Image Size | 640x640 |
    | Batch Size | 16 |
    | Training Images | 2,850 |
    """)
with col2:
    st.markdown("""
    | Dataset Info | Value |
    |-------------|-------|
    | Total Images | 1,188 |
    | Augmented Total | 2,850 |
    | Train Set | 831 images |
    | Validation Set | 236 images |
    | Test Set | 121 images |
    """)

st.markdown("---")
st.subheader("About FireGuard AI")

st.markdown("""
### 🎯 Problem Statement
Manual fire safety inspections are **costly, infrequent, and error-prone**.
Buildings in Germany and worldwide are legally required to maintain fire 
extinguishers at certified locations — but compliance is rarely monitored in real-time.

### 💡 Solution
FireGuard AI uses **YOLOv8 computer vision** to automatically detect whether 
fire extinguishers are present in any room or corridor — enabling continuous, 
automated safety compliance monitoring.

### 🏢 Business Value
- 🔴 Reduces manual inspection costs by up to **70%**
- ⚡ Enables **24/7 automated** compliance monitoring  
- 🚨 Instant alerts for non-compliant zones
- 📊 Audit trail for safety inspections

### 🛠️ Tech Stack
- **Model:** YOLOv8s (Ultralytics)
- **Dataset:** 1,188 images + augmentation = 2,850 training images
- **Framework:** Python, Streamlit
- **Training:** Google Colab (T4 GPU)

### 👨‍💻 Developed by
**Pratik Yadav** | Vision Technology Internship | SJCEM 2026
""")

# Footer
st.markdown("---")
st.markdown(
    "<center style='color:#94a3b8;'>Built with ❤️ by Pratik Yadav | FireGuard AI | YOLOv8 Powered</center>",
    unsafe_allow_html=True
)