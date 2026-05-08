import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="FireGuard AI",
    page_icon="🧯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### FireGuard AI\nAutomated Fire Extinguisher Detection System"
    }
)

# Enhanced CSS with animations and better styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
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
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }

    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(255, 91, 95, 0.2);
        }
        50% {
            box-shadow: 0 0 30px rgba(255, 91, 95, 0.4);
        }
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 91, 95, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(255, 154, 90, 0.12), transparent 22%),
            linear-gradient(180deg, #08101f 0%, #0b1327 42%, #09101d 100%);
        color: var(--text);
        animation: fadeIn 0.6s ease-in;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem 2rem 1.5rem;
        border: 1px solid var(--panel-border);
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(13, 18, 34, 0.78));
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
        animation: slideInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--text);
        text-align: center;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--text), var(--accent-2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--muted);
        text-align: center;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }

    .kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0 0.5rem;
    }

    .kpi {
        padding: 1.2rem;
        border-radius: 20px;
        border: 1px solid var(--panel-border);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
        backdrop-filter: blur(10px);
        transition: var(--transition);
        cursor: pointer;
        animation: slideInUp 0.6s ease-out;
    }

    .kpi:hover {
        border-color: rgba(255, 255, 255, 0.15);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.04));
        transform: translateY(-4px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--accent-2);
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .kpi-delta {
        margin-top: 0.4rem;
        font-size: 0.82rem;
        color: #cbd5e1;
    }

    .panel {
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(20, 30, 50, 0.6), rgba(10, 15, 30, 0.4));
        padding: 1.4rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        transition: var(--transition);
        animation: slideInUp 0.6s ease-out;
    }

    .panel:hover {
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
    }

    .panel h3 {
        color: var(--text);
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .panel-label {
        color: var(--accent-2);
        font-size: 1rem;
        margin-bottom: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .compliant {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(45, 212, 191, 0.15));
        border: 2px solid rgba(45, 212, 191, 0.4);
        padding: 1.3rem 1.3rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
        color: #a7f3d0;
        margin-bottom: 1.2rem;
        animation: slideInUp 0.6s ease-out;
    }
    
    .non-compliant {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.2), rgba(255, 154, 90, 0.15));
        border: 2px solid rgba(248, 113, 113, 0.4);
        padding: 1.3rem 1.3rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffa8a8;
        margin-bottom: 1.2rem;
        animation: slideInUp 0.6s ease-out;
        animation-delay: 0.1s;
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.5rem;
    }

    .dash-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 1.1rem;
        transition: var(--transition);
    }

    .dash-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.04));
        transform: translateY(-3px);
    }

    .dash-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    .dash-value {
        color: var(--accent-2);
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .dash-note {
        color: #cbd5e1;
        font-size: 0.84rem;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    .confidence-box {
        margin: 0.8rem 0 1rem;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(255, 154, 90, 0.3);
        background: rgba(255, 154, 90, 0.08);
        color: #ffe0a8;
        font-weight: 600;
        transition: var(--transition);
    }

    .confidence-box:hover {
        border-color: rgba(255, 154, 90, 0.5);
        background: rgba(255, 154, 90, 0.12);
    }

    .confidence-box strong {
        color: var(--accent-2);
        font-weight: 800;
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin: 0.8rem 0 1rem;
    }

    .mini-status {
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        color: #dbe7ff;
        font-size: 0.85rem;
        font-weight: 700;
        transition: var(--transition);
    }

    .mini-status:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.15);
    }

    .mini-status strong {
        color: var(--accent-2);
    }

    .result-shell {
        margin-top: 1rem;
        padding: 1.3rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.88), rgba(12, 17, 31, 0.96));
        border: 1px solid var(--panel-border);
        animation: slideInUp 0.6s ease-out;
        animation-delay: 0.1s;
    }

    .image-frame {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        transition: var(--transition);
    }

    .image-frame:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
        color: #fff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1.5rem !important;
        border-radius: 12px !important;
        transition: var(--transition) !important;
        box-shadow: 0 8px 20px rgba(255, 91, 95, 0.3) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(255, 91, 95, 0.5) !important;
    }

    .stButton>button:active {
        transform: translateY(0) !important;
    }

    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 2px dashed rgba(255, 154, 90, 0.3) !important;
        border-radius: 18px !important;
        padding: 0.5rem !important;
        transition: var(--transition) !important;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(255, 154, 90, 0.6) !important;
        background: rgba(255, 154, 90, 0.05) !important;
    }

    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 12, 24, 0.98), rgba(13, 18, 34, 0.99)) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] .sidebar-card {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        transition: var(--transition) !important;
        animation: slideInUp 0.6s ease-out;
    }

    [data-testid="stSidebar"] .sidebar-card:hover {
        border-color: rgba(255, 255, 255, 0.12) !important;
        background: rgba(255, 255, 255, 0.06) !important;
    }

    /* Table styling */
    table {
        background: rgba(10, 14, 22, 0.95) !important;
        border-collapse: collapse !important;
    }

    table thead th {
        background: linear-gradient(135deg, rgba(30, 41, 70, 0.98), rgba(20, 30, 50, 0.98)) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        border-bottom: 2px solid rgba(255, 154, 90, 0.3) !important;
        padding: 12px 16px !important;
    }

    table tbody td {
        background: rgba(14, 20, 39, 0.9) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 12px 16px !important;
        transition: var(--transition) !important;
    }

    table tbody tr:hover td {
        background: rgba(35, 48, 80, 0.98) !important;
        color: #ffffff !important;
    }

    /* Metrics styling */
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        transition: var(--transition) !important;
    }

    .stMetric:hover {
        border-color: rgba(255, 255, 255, 0.15) !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03)) !important;
    }

    /* Tags */
    .tag {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }

    .tag-success {
        background: rgba(45, 212, 191, 0.18);
        color: #a7f3d0;
        border: 1px solid rgba(45, 212, 191, 0.35);
    }

    .tag-danger {
        background: rgba(248, 113, 113, 0.18);
        color: #ffa8a8;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    .tag-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #dbe7ff;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .kpi-row {
            grid-template-columns: 1fr;
        }
        .dashboard-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Load model
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### 🧯 FireGuard AI")
    st.markdown("""
    <div class='sidebar-card'>
        <div class='subtle-text'>Automated fire extinguisher detection system for compliance audits.</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        1. **Upload Image** - Upload a clear room/corridor photo
        2. **Set Threshold** - Adjust confidence level (default: 0.5)
        3. **Analyze** - System detects fire extinguishers
        4. **Review Results** - Check compliance status
        """)
    
    with st.expander("⚙️ Threshold Guide"):
        st.markdown("""
        - **0.1-0.3**: Very sensitive (more false positives)
        - **0.4-0.6**: Balanced (recommended)
        - **0.7-1.0**: Strict (fewer detections)
        """)
    
    with st.expander("💡 Pro Tips"):
        st.markdown("""
        - Use well-lit, clear images
        - Ensure extinguisher is visible
        - Avoid blur or occlusion
        - Best aspect ratio: 4:3 or 16:9
        """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "YOLOv8s", "Custom Trained")
    with col2:
        st.metric("Precision", "98%", "+18%")
    with col3:
        st.metric("Recall", "95%", "+17%")

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
                <div class="kpi-delta">Trained on 2,850 images</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Use Case</div>
                <div class="kpi-value">Safety Audit</div>
                <div class="kpi-delta">Room & corridor inspection</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Status</div>
                <div class="kpi-value">Ready</div>
                <div class="kpi-delta">Upload image to detect</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main content area
col1, col2 = st.columns([1.1, 1.3], gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">📋 Control Panel</div>', unsafe_allow_html=True)
    
    st.subheader("Image Upload", divider=True)
    uploaded = st.file_uploader(
        "Select an image file",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
        key="file_uploader"
    )
    st.caption("Supported formats: JPG, JPEG, PNG (Max: 200MB)")
    
    st.subheader("Detection Settings", divider=True)
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="How confident the model must be to make a detection (0.1=sensitive, 1.0=strict)"
    )
    
    threshold_level = "🟢 Sensitive" if confidence < 0.4 else "🟡 Balanced" if confidence < 0.7 else "🔴 Strict"
    st.markdown(
        f'<div class="confidence-box"><strong>{threshold_level}</strong> - Threshold: <strong>{confidence:.2f}</strong></div>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    if uploaded:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'<div class="mini-status">Size: <strong>{uploaded.size / 1024:.1f}KB</strong></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="mini-status">Type: <strong>{uploaded.type}</strong></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">📊 Detection Results</div>', unsafe_allow_html=True)
    
    if uploaded is not None:
        try:
            # Load and analyze image
            image = Image.open(uploaded)
            
            # Display with progress
            with st.spinner("🔍 Analyzing image..."):
                time.sleep(0.3)  # Visual feedback
                results = model(image, conf=confidence)
            
            count = len(results[0].boxes)
            top_conf = float(results[0].boxes.conf.max()) if count > 0 else 0.0
            compliance = "COMPLIANT ✅" if count > 0 else "NON-COMPLIANT ❌"
            compliance_class = "compliant" if count > 0 else "non-compliant"
            
            # Compliance status
            st.markdown(
                f'<div class="{compliance_class}"><strong>{compliance}</strong><br>{count} Fire Extinguisher(s) Detected</div>',
                unsafe_allow_html=True
            )
            
            # Key metrics
            st.markdown(
                f"""
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <div class="dash-label">Detections</div>
                        <div class="dash-value">{count}</div>
                        <div class="dash-note">Objects found</div>
                    </div>
                    <div class="dash-card">
                        <div class="dash-label">Top Confidence</div>
                        <div class="dash-value">{top_conf:.0%}</div>
                        <div class="dash-note">Highest score</div>
                    </div>
                    <div class="dash-card">
                        <div class="dash-label">Threshold</div>
                        <div class="dash-value">{confidence:.2f}</div>
                        <div class="dash-note">Filter setting</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Image tabs
            tab1, tab2 = st.tabs(["📸 Detection Result", "📷 Original Image"])
            
            with tab1:
                st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                result_image = results[0].plot()
                result_image_rgb = result_image[..., ::-1]  # Convert BGR to RGB
                st.image(result_image_rgb, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                st.image(image, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detection details
            st.markdown('<div class="result-shell">', unsafe_allow_html=True)
            st.subheader("Detection Details", divider=True)
            
            if count > 0:
                detection_data = []
                for i, box in enumerate(results[0].boxes):
                    conf_score = float(box.conf[0])
                    detection_data.append({
                        "🎯 Detection": f"Fire Extinguisher {i+1}",
                        "📊 Confidence": f"{conf_score:.1%}",
                        "Status": "✅ Valid" if conf_score >= confidence else "❌ Below Threshold"
                    })
                
                st.dataframe(detection_data, use_container_width=True, hide_index=True)
                
                st.success(f"✅ Compliance Check Passed - {count} fire extinguisher(s) detected!")
            else:
                st.warning("⚠️ No fire extinguishers detected - Room is NON-COMPLIANT")
                st.info("Try adjusting the confidence threshold or uploading a clearer image.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.session_state.analysis_complete = True
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("Please try uploading a different image or contact support.")
    else:
        st.markdown(
            '<div class="tag tag-danger">No image uploaded yet</div>',
            unsafe_allow_html=True
        )
        st.info("📤 Upload an image on the left side to start detection!")
        
        # Placeholder
        placeholder_w, placeholder_h = 900, 520
        placeholder = Image.new("RGB", (placeholder_w, placeholder_h), "#101826")
        draw = ImageDraw.Draw(placeholder)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        text = "📤 Upload an Image to Begin"
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            text_w, text_h = (len(text) * 10, 20)
        draw.text(((placeholder_w - text_w) / 2, (placeholder_h - text_h) / 2), text, fill="#edf2ff", font=font)
        st.image(placeholder, use_column_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Model Performance Section
st.markdown("### 📈 Model Performance Metrics")

perf1, perf2, perf3, perf4 = st.columns(4)
with perf1:
    st.metric(label="🎯 Precision", value="98%", delta="+18% vs target")
with perf2:
    st.metric(label="🔍 Recall", value="95%", delta="+17% vs target")
with perf3:
    st.metric(label="📊 mAP@50", value="95%", delta="+13% vs target")
with perf4:
    st.metric(label="📊 mAP@50-95", value="80%", delta="+15% vs target")

st.divider()

# Training Details Section
st.markdown("### 🛠️ Training Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Model Parameters**
    | Parameter | Value |
    |-----------|-------|
    | Architecture | YOLOv8s |
    | Epochs | 50 |
    | Image Size | 640×640 |
    | Batch Size | 16 |
    | Optimizer | SGD |
    """)

with col2:
    st.markdown("""
    **Dataset Composition**
    | Dataset | Count |
    |---------|-------|
    | Original Images | 1,188 |
    | Augmented Total | 2,850 |
    | Training Set | 831 (71%) |
    | Validation Set | 236 (20%) |
    | Test Set | 121 (9%) |
    """)

with col3:
    st.markdown("""
    **Augmentation Techniques**
    | Technique | Status |
    |-----------|--------|
    | Rotation | ✅ |
    | Flip | ✅ |
    | Brightness | ✅ |
    | Contrast | ✅ |
    | Scale | ✅ |
    """)

st.divider()

# About Section
st.markdown("### 🎯 About FireGuard AI")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🏢 Use Cases", "💼 Business Value", "👨‍💻 Technical Stack"])

with tab1:
    st.markdown("""
    #### Problem Statement
    
    Manual fire safety inspections are **costly, infrequent, and error-prone**. 
    Buildings in Germany and worldwide are legally required to maintain fire extinguishers 
    at certified locations — but compliance is rarely monitored in real-time.
    
    #### Solution
    
    FireGuard AI uses **YOLOv8 computer vision** to automatically detect whether fire extinguishers 
    are present in any room or corridor — enabling continuous, automated safety compliance monitoring.
    
    #### Key Features
    - 🤖 **AI-Powered Detection** - Real-time fire extinguisher identification
    - ⚡ **Fast Processing** - Analyze images in milliseconds
    - 🎯 **High Accuracy** - 98% precision, 95% recall
    - 🔧 **Adjustable Sensitivity** - Customizable confidence thresholds
    """)

with tab2:
    st.markdown("""
    #### Primary Use Cases
    
    1. **Facility Management** - Regular compliance audits
    2. **Insurance Verification** - Safety certification
    3. **Security Patrols** - Automated monitoring
    4. **Building Inspections** - Real-time reporting
    5. **Safety Training** - Identification practice
    
    #### Industry Applications
    - 🏢 Office Buildings
    - 🏭 Manufacturing Plants
    - 🏨 Hotels & Hospitality
    - 🏥 Hospitals & Clinics
    - 🛒 Retail Stores
    - 📚 Educational Institutions
    """)

with tab3:
    st.markdown("""
    #### Business Impact
    
    | Metric | Value | Impact |
    |--------|-------|--------|
    | Cost Reduction | 70% | Lower inspection costs |
    | Monitoring | 24/7 | Continuous coverage |
    | Detection Speed | <1 sec | Instant feedback |
    | Accuracy | 98% | Reliable results |
    | Compliance | Auto | Audit trail ready |
    
    #### ROI Benefits
    - 🔴 **Reduce Liability** - Ensure compliance automatically
    - ⚡ **Improve Efficiency** - Automate manual processes
    - 🚨 **Instant Alerts** - Get non-compliance notifications
    - 📊 **Data Insights** - Generate compliance reports
    """)

with tab4:
    st.markdown("""
    #### Technology Stack
    
    **Computer Vision**
    - Framework: YOLOv8 (Ultralytics)
    - Model: YOLOv8s (Small variant)
    - Inference: ONNX Runtime
    
    **Web Application**
    - Frontend: Streamlit
    - Backend: Python 3.11+
    - Deployment: Streamlit Cloud
    
    **Development & Training**
    - Training: Google Colab (NVIDIA T4 GPU)
    - Dataset Tool: Roboflow
    - Version Control: Git & GitHub
    - Libraries: PyTorch, NumPy, Pillow, OpenCV
    
    #### Performance Specifications
    - Inference Time: ~50-100ms per image
    - Memory Usage: ~300MB
    - Model Size: ~26MB
    - GPU Memory: ~500MB
    """)

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08);">
    <p style="margin: 0; font-weight: 600;">🧯 FireGuard AI</p>
    <p style="margin: 0.5rem 0 0; font-size: 0.9rem;">Built with ❤️ for Fire Safety Compliance</p>
    <p style="margin: 1rem 0 0; font-size: 0.85rem;">YOLOv8 Powered | Streamlit Hosted | © 2024</p>
</div>
""", unsafe_allow_html=True)