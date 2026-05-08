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

# Professional Dark Mode CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    :root {
        --bg-primary: #05070d;
        --bg-secondary: #0a0f1a;
        --bg-tertiary: #0f1420;
        --panel-dark: rgba(10, 15, 28, 0.95);
        --panel-border: rgba(255, 91, 95, 0.15);
        --text-primary: #ffffff;
        --text-secondary: #b0b8c8;
        --accent: #ff4757;
        --accent-warm: #ffa502;
        --success: #2ed573;
        --danger: #ff4757;
        --transition: all 0.2s ease;
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(255, 71, 87, 0.1);
        }
        50% {
            box-shadow: 0 0 40px rgba(255, 71, 87, 0.2);
        }
    }

    .stApp {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 50%, #0f1420 100%);
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Hero Section */
    .hero {
        padding: 2.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 20, 40, 0.95) 100%);
        border: 1.5px solid var(--panel-border);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        animation: slideInUp 0.6s ease-out;
    }

    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: var(--text-primary);
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(255, 71, 87, 0.2);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    .kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.2rem;
        margin: 1.5rem 0;
    }

    .kpi {
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 71, 87, 0.1);
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.6) 0%, rgba(15, 22, 40, 0.4) 100%);
        backdrop-filter: blur(8px);
        transition: var(--transition);
        animation: slideInUp 0.6s ease-out;
    }

    .kpi:hover {
        border-color: rgba(255, 255, 255, 0.15);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.04));
        transform: translateY(-4px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .kpi:hover {
        border-color: rgba(255, 71, 87, 0.3);
        background: linear-gradient(135deg, rgba(25, 35, 60, 0.8) 0%, rgba(20, 28, 48, 0.6) 100%);
        transform: translateY(-4px);
    }

    .kpi-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        margin-bottom: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--accent-warm);
        line-height: 1;
        margin-bottom: 0.4rem;
    }

    .kpi-delta {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    /* Panels */
    .panel {
        border: 1.5px solid var(--panel-border);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15, 22, 42, 0.9) 0%, rgba(10, 15, 30, 0.95) 100%);
        padding: 1.8rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: var(--transition);
        animation: slideInUp 0.6s ease-out;
    }

    .panel:hover {
        border-color: rgba(255, 71, 87, 0.25);
        box-shadow: 0 25px 70px rgba(255, 71, 87, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .panel h3 {
        color: var(--text-primary);
        margin: 0 0 1.2rem 0;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .panel-label {
        color: var(--accent-warm);
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Compliance Status */
    .compliant {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.15) 0%, rgba(30, 160, 85, 0.1) 100%);
        border: 2px solid rgba(46, 213, 115, 0.4);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        color: #4ade80;
        margin-bottom: 1.5rem;
        animation: slideInUp 0.6s ease-out;
        letter-spacing: -0.01em;
    }
    
    .non-compliant {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 2px solid rgba(255, 71, 87, 0.4);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        color: #ff6b7a;
        margin-bottom: 1.5rem;
        animation: slideInUp 0.6s ease-out;
        letter-spacing: -0.01em;
    }

    /* Dashboard Grid */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1.2rem;
        margin: 1.3rem 0;
    }

    .dash-card {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.6) 0%, rgba(15, 20, 40, 0.4) 100%);
        border: 1.5px solid rgba(255, 71, 87, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        transition: var(--transition);
        animation: slideInUp 0.6s ease-out;
    }

    .dash-card:hover {
        border-color: rgba(255, 71, 87, 0.25);
        background: linear-gradient(135deg, rgba(25, 35, 60, 0.75) 0%, rgba(20, 28, 48, 0.5) 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255, 71, 87, 0.1);
    }

    .dash-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
    }

    .dash-value {
        color: var(--accent-warm);
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .dash-note {
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* Confidence Box */
    .confidence-box {
        margin: 1rem 0;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1.5px solid rgba(255, 165, 2, 0.3);
        background: linear-gradient(135deg, rgba(255, 165, 2, 0.1) 0%, rgba(255, 165, 2, 0.05) 100%);
        color: #fbbf24;
        font-weight: 600;
        transition: var(--transition);
    }

    .confidence-box:hover {
        border-color: rgba(255, 165, 2, 0.5);
        background: linear-gradient(135deg, rgba(255, 165, 2, 0.15) 0%, rgba(255, 165, 2, 0.08) 100%);
    }

    .confidence-box strong {
        color: var(--accent-warm);
        font-weight: 800;
    }

    /* Status Row */
    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .mini-status {
        padding: 0.6rem 1rem;
        border-radius: 20px;
        border: 1.5px solid rgba(255, 71, 87, 0.15);
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.6) 0%, rgba(15, 20, 40, 0.4) 100%);
        color: var(--text-primary);
        font-size: 0.85rem;
        font-weight: 700;
        transition: var(--transition);
    }

    .mini-status:hover {
        background: linear-gradient(135deg, rgba(25, 35, 60, 0.75) 0%, rgba(20, 28, 48, 0.5) 100%);
        border-color: rgba(255, 71, 87, 0.3);
    }

    .mini-status strong {
        color: var(--accent-warm);
    }

    /* Result Shell */
    .result-shell {
        margin-top: 1.2rem;
        padding: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15, 22, 42, 0.9) 0%, rgba(10, 15, 30, 0.95) 100%);
        border: 1.5px solid var(--panel-border);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: slideInUp 0.6s ease-out;
    }

    /* Image Frame */
    .image-frame {
        border-radius: 18px;
        overflow: hidden;
        border: 1.5px solid rgba(255, 71, 87, 0.15);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
        transition: var(--transition);
    }

    .image-frame:hover {
        border-color: rgba(255, 71, 87, 0.3);
        box-shadow: 0 25px 60px rgba(255, 71, 87, 0.1);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-warm) 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1.8rem !important;
        border-radius: 12px !important;
        transition: var(--transition) !important;
        box-shadow: 0 8px 24px rgba(255, 71, 87, 0.3) !important;
        letter-spacing: 0.01em !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(255, 71, 87, 0.5) !important;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(15, 22, 42, 0.8) 0%, rgba(10, 15, 30, 0.9) 100%) !important;
        border: 2px dashed rgba(255, 71, 87, 0.25) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        transition: var(--transition) !important;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(255, 71, 87, 0.5) !important;
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 22, 42, 0.95) 100%) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #05070d 0%, #0a0f1a 100%) !important;
        border-right: 1.5px solid rgba(255, 71, 87, 0.1) !important;
    }

    /* Tables */
    table {
        background: rgba(10, 15, 30, 0.95) !important;
        border-collapse: collapse !important;
    }

    table thead th {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.95) 0%, rgba(15, 20, 40, 0.95) 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border-bottom: 2px solid rgba(255, 71, 87, 0.2) !important;
        padding: 12px 16px !important;
    }

    table tbody td {
        background: rgba(10, 15, 30, 0.8) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(255, 71, 87, 0.08) !important;
        padding: 12px 16px !important;
    }

    table tbody tr:hover td {
        background: rgba(20, 27, 50, 0.95) !important;
    }

    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.6) 0%, rgba(15, 20, 40, 0.4) 100%) !important;
        border: 1.5px solid rgba(255, 71, 87, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        transition: var(--transition) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }

    .stMetric:hover {
        border-color: rgba(255, 71, 87, 0.25) !important;
        background: linear-gradient(135deg, rgba(25, 35, 60, 0.75) 0%, rgba(20, 28, 48, 0.5) 100%) !important;
    }

    /* Tags */
    .tag {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.01em;
        margin-bottom: 0.8rem;
    }

    .tag-success {
        background: rgba(46, 213, 115, 0.15);
        color: #4ade80;
        border: 1px solid rgba(46, 213, 115, 0.3);
    }

    .tag-danger {
        background: rgba(255, 71, 87, 0.15);
        color: #ff6b7a;
        border: 1px solid rgba(255, 71, 87, 0.3);
    }

    .tag-neutral {
        background: rgba(176, 184, 200, 0.1);
        color: var(--text-secondary);
        border: 1px solid rgba(176, 184, 200, 0.2);
    }

    /* Subtle Text */
    .subtle-text {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        .kpi-row, .dashboard-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Fix all Streamlit white patches */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 50%, #0f1420 100%) !important;
    }

    [data-testid="stVerticalBlock"] {
        background: transparent !important;
    }

    /* Input fields and text areas */
    input {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.8) 0%, rgba(15, 20, 40, 0.8) 100%) !important;
        color: white !important;
        border: 1.5px solid rgba(255, 71, 87, 0.15) !important;
        border-radius: 12px !important;
    }

    input:focus {
        border-color: rgba(255, 71, 87, 0.35) !important;
        box-shadow: 0 0 0 3px rgba(255, 71, 87, 0.1) !important;
    }

    textarea {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.8) 0%, rgba(15, 20, 40, 0.8) 100%) !important;
        color: white !important;
        border: 1.5px solid rgba(255, 71, 87, 0.15) !important;
        border-radius: 12px !important;
    }

    textarea:focus {
        border-color: rgba(255, 71, 87, 0.35) !important;
        box-shadow: 0 0 0 3px rgba(255, 71, 87, 0.1) !important;
    }

    select {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.8) 0%, rgba(15, 20, 40, 0.8) 100%) !important;
        color: white !important;
        border: 1.5px solid rgba(255, 71, 87, 0.15) !important;
        border-radius: 12px !important;
    }

    /* Streamlit dividers */
    [data-testid="stHorizontalBlock"] {
        background: transparent !important;
    }

    hr {
        border-color: rgba(255, 71, 87, 0.1) !important;
        margin: 2rem 0 !important;
    }

    /* Hide white backgrounds */
    .stTabs, [data-testid="stTabs"] {
        background: transparent !important;
    }

    /* File uploader fix */
    [data-testid="stFileUploader"] section {
        background: linear-gradient(135deg, rgba(15, 22, 42, 0.9) 0%, rgba(10, 15, 30, 0.95) 100%) !important;
        border: 2px dashed rgba(255, 71, 87, 0.25) !important;
        border-radius: 16px !important;
    }

    [data-testid="stFileUploader"] section:hover {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.95) 0%, rgba(15, 22, 42, 0.98) 100%) !important;
        border-color: rgba(255, 71, 87, 0.4) !important;
    }

    /* Hide all white text inputs */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.8) 0%, rgba(15, 20, 40, 0.8) 100%) !important;
        color: white !important;
        caret-color: #ffa502 !important;
    }

    /* Streamlit columns */
    [data-testid="column"] {
        background: transparent !important;
    }

    /* All text colors */
    p, span, label, h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* Expander styling */
    [data-testid="stExpander"] {
        background: transparent !important;
        border: 1.5px solid rgba(255, 71, 87, 0.1) !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] > div {
        background: transparent !important;
    }

    /* Tab styling */
    [data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(255, 71, 87, 0.15) !important;
        border-bottom: 2px solid rgba(255, 71, 87, 0.5) !important;
    }

    [data-testid="stTabs"] [aria-selected="false"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255, 71, 87, 0.1) !important;
    }

    /* Number input */
    .stNumberInput > div > div > input {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.8) 0%, rgba(15, 20, 40, 0.8) 100%) !important;
        color: white !important;
        caret-color: #ffa502 !important;
        border: 1.5px solid rgba(255, 71, 87, 0.15) !important;
    }

    /* Select box */
    [data-testid="stSelectbox"] > div > div > div {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.8) 0%, rgba(15, 20, 40, 0.8) 100%) !important;
        border: 1.5px solid rgba(255, 71, 87, 0.15) !important;
        color: white !important;
    }

    /* Slider styling */
    [data-testid="stSlider"] {
        background: transparent !important;
    }

    [data-testid="stSlider"] input {
        accent-color: #ff4757 !important;
    }

    /* Checkbox styling */
    [data-testid="stCheckbox"] {
        background: transparent !important;
    }

    [data-testid="stCheckbox"] input {
        accent-color: #ff4757 !important;
    }

    /* Radio styling */
    [data-testid="stRadio"] {
        background: transparent !important;
    }

    [data-testid="stRadio"] input {
        accent-color: #ff4757 !important;
    }

    /* Data frame styling */
    [data-testid="stDataFrame"] {
        background: rgba(10, 15, 30, 0.95) !important;
    }

    .streamlit-expanderContent {
        background: rgba(10, 15, 30, 0.5) !important;
    }

    /* Code block */
    pre {
        background: rgba(10, 15, 30, 0.95) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 71, 87, 0.1) !important;
    }

    code {
        background: rgba(20, 27, 50, 0.6) !important;
        color: #ffa502 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Info/Warning/Error boxes */
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 20, 40, 0.95) 100%) !important;
        border: 1.5px solid rgba(255, 71, 87, 0.2) !important;
        border-left: 4px solid #ff4757 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Remove any remaining white backgrounds */
    [style*="white"], [style*="#fff"], [style*="#ffffff"] {
        filter: invert(1) !important;
    }

    /* Success message styling */
    .streamlit-alert-success {
        border-left-color: #2ed573 !important;
    }

    /* Error message styling */
    .streamlit-alert-danger, .streamlit-alert-error {
        border-left-color: #ff4757 !important;
    }

    /* Warning styling */
    .streamlit-alert-warning {
        border-left-color: #ffa502 !important;
    }

    /* Loading spinner */
    [data-testid="stSpinner"] {
        color: #ff4757 !important;
    }

    /* Progress bar */
    [data-testid="stProgress"] {
        background: rgba(20, 27, 50, 0.6) !important;
    }

    /* Ensure all visible text is white or light colored */
    .stMarkdown, [data-testid="stMarkdownContainer"] {
        color: white !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: white !important;
    }

    /* Link styling */
    a {
        color: #ffa502 !important;
        text-decoration: none !important;
    }

    a:hover {
        color: #ff4757 !important;
        text-decoration: underline !important;
    }

    /* Caption and small text */
    .streamlit-caption, [data-testid="stCaption"] {
        color: #b0b8c8 !important;
    }

    /* Block container background */
    .main, .block-container {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 50%, #0f1420 100%) !important;
    }

    /* Remove all light backgrounds from containers */
    .stContainer {
        background: transparent !important;
    }

    /* ===== AGGRESSIVE WHITE PATCH FIXES ===== */
    
    /* Top header/toolbar area */
    header {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 100%) !important;
        border-bottom: 1px solid rgba(255, 71, 87, 0.1) !important;
    }

    [data-testid="stHeader"] {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 100%) !important;
    }

    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 100%) !important;
        border-right: 1.5px solid rgba(255, 71, 87, 0.1) !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background: transparent !important;
    }

    /* Fix metric cards that show white */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.8) 0%, rgba(15, 20, 40, 0.8) 100%) !important;
        border: 1.5px solid rgba(255, 71, 87, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }

    [data-testid="stMetric"] > div {
        background: transparent !important;
    }

    [data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        background: transparent !important;
    }

    /* Metric text colors */
    [data-testid="stMetric"] label {
        color: #b0b8c8 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
    }

    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffa502 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* Column backgrounds */
    [data-testid="stColumn"] {
        background: transparent !important;
    }

    [data-testid="stColumn"] > div {
        background: transparent !important;
    }

    /* Horizontal block */
    [data-testid="stHorizontalBlock"] {
        background: transparent !important;
    }

    [data-testid="stHorizontalBlock"] > div {
        background: transparent !important;
    }

    /* Vertical blocks */
    [data-testid="stVerticalBlock"] {
        background: transparent !important;
    }

    [data-testid="stVerticalBlock"] > div {
        background: transparent !important;
    }

    /* Expand/collapse sections */
    [data-testid="stExpanderContent"] {
        background: rgba(15, 22, 42, 0.8) !important;
        border: 1px solid rgba(255, 71, 87, 0.1) !important;
        border-top: 1.5px solid rgba(255, 71, 87, 0.15) !important;
    }

    /* Fix button container backgrounds */
    [data-testid="stElementContainer"] {
        background: transparent !important;
    }

    /* Ensure upload button area is dark */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] > div {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] > div > div {
        background: linear-gradient(135deg, rgba(15, 22, 42, 0.9) 0%, rgba(10, 15, 30, 0.95) 100%) !important;
        border: 2px dashed rgba(255, 71, 87, 0.25) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }

    [data-testid="stFileUploader"] > div > div:hover {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.95) 0%, rgba(15, 22, 42, 0.98) 100%) !important;
        border-color: rgba(255, 71, 87, 0.4) !important;
    }

    /* Button styling - fix upload button */
    button, [role="button"] {
        background: linear-gradient(135deg, #ff4757 0%, #ffa502 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.8rem 1.8rem !important;
        box-shadow: 0 8px 24px rgba(255, 71, 87, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    button:hover, [role="button"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(255, 71, 87, 0.5) !important;
    }

    button:active, [role="button"]:active {
        transform: translateY(0) !important;
    }

    /* Subheader text */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* Divider line color */
    hr, [data-testid="stHorizontalBlock"] {
        border-color: rgba(255, 71, 87, 0.1) !important;
        margin: 2rem 0 !important;
    }

    /* Text content styling */
    p {
        color: white !important;
    }

    span {
        color: white !important;
    }

    label {
        color: white !important;
    }

    /* Markdown content */
    [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        color: white !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: white !important;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: white !important;
    }

    /* Caption text */
    [data-testid="stCaption"] {
        color: #b0b8c8 !important;
    }

    .streamlit-caption {
        color: #b0b8c8 !important;
    }

    /* Small text styling */
    small {
        color: #b0b8c8 !important;
    }

    /* Alert boxes - dark themed */
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 20, 40, 0.95) 100%) !important;
        border: 1.5px solid rgba(255, 71, 87, 0.2) !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
    }

    [data-testid="stAlert"] > div {
        color: white !important;
    }

    /* Info message */
    .stAlert-info {
        border-left: 4px solid #2ed573 !important;
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 20, 40, 0.95) 100%) !important;
    }

    /* Success message */
    .stAlert-success {
        border-left: 4px solid #2ed573 !important;
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 20, 40, 0.95) 100%) !important;
    }

    /* Warning message */
    .stAlert-warning {
        border-left: 4px solid #ffa502 !important;
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 20, 40, 0.95) 100%) !important;
    }

    /* Error message */
    .stAlert-error {
        border-left: 4px solid #ff4757 !important;
        background: linear-gradient(135deg, rgba(20, 27, 50, 0.9) 0%, rgba(15, 20, 40, 0.95) 100%) !important;
    }

    /* Subheader line */
    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, transparent, rgba(255, 71, 87, 0.2), transparent) !important;
        height: 2px !important;
    }

    /* Fix metric numbers and labels */
    [data-testid="stMetric"] > div > div:first-child {
        background: transparent !important;
    }

    /* Ensure all list items are properly styled */
    li {
        color: white !important;
    }

    /* Code styling */
    code {
        background: rgba(20, 27, 50, 0.8) !important;
        color: #ffa502 !important;
        border: 1px solid rgba(255, 71, 87, 0.1) !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }

    pre {
        background: rgba(10, 15, 30, 0.95) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 71, 87, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Ensure body text is never light */
    body {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 50%, #0f1420 100%) !important;
        color: white !important;
    }

    html {
        background: linear-gradient(135deg, #05070d 0%, #0a0f1a 50%, #0f1420 100%) !important;
    }

    /* Force all unspecified text to be white or light */
    * {
        color: inherit !important;
    }

    /* Link styling */
    a {
        color: #ffa502 !important;
    }

    a:hover {
        color: #ff4757 !important;
    }

    /* Selection colors */
    ::selection {
        background: rgba(255, 71, 87, 0.3) !important;
        color: white !important;
    }

    ::-moz-selection {
        background: rgba(255, 71, 87, 0.3) !important;
        color: white !important;
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