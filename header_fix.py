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

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
