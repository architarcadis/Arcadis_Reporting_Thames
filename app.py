# Thames Water Enterprise Analytics Platform
# Main Streamlit Application Entry Point

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import uuid
import base64
from fastapi import FastAPI, Depends
import time

# Import components
from components.data_quality import run_data_quality_check
from components.insights import generate_insights, generate_action_recommendations
from components.financial import calculate_roi, calculate_financial_impact
from components.visualization import create_trend_chart, create_kpi_indicator
from components.notifications import prioritize_alerts
from components.whitelist import apply_client_theme
from app_theme import fix_all_styling

# Set page config with Arcadis theming
st.set_page_config(
    page_title="Arcadis Water Utility Analytics Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Embed CSS directly instead of loading from files
# This fixes deployment issues on Streamlit Cloud
st.markdown('''
<style>
/* Custom CSS for RiskLensPro-inspired design */

/* Main page background */
.main {
    background-color: #F8F9FA;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #2C3E50;
}

.sidebar .sidebar-content * {
    color: white;
}

/* Tab styling - forceful override */
div[data-testid="stTabs"] > div > div > div > div[role="tablist"] {
    background-color: transparent;
}

div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button {
    background-color: #f0f0f0;
    color: #333333 !important;
    border-radius: 5px 5px 0 0;
    padding: 10px 16px;
    font-weight: 500;
    border: 1px solid #e0e0e0;
    border-bottom: none;
    margin-right: 4px;
}

div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button:hover {
    background-color: #e0e0e0;
    color: #333333;
}

div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button[aria-selected="true"] {
    background-color: #FF6900;
    color: white !important;
    font-weight: 600;
    border: 1px solid #FF6900;
}

div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button p {
    color: inherit !important;
}

/* Cards styling */
div.stCard {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 1rem;
    transition: transform 0.3s, box-shadow 0.3s;
}

div.stCard:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.06);
}

/* Button styling */
button, div.stButton button, div.stDownloadButton button {
    background-color: #FF6900 !important;
    color: white !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

button:hover, div.stButton button:hover, div.stDownloadButton button:hover {
    background-color: #E65800 !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    transform: translateY(-2px) !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

div[data-testid="stMetric"] label {
    color: #2C3E50 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #FF6900 !important;
    font-weight: 700 !important;
}

/* Headers styling */
h1, h2, h3 {
    color: #2C3E50 !important;
    font-weight: 600 !important;
}

h1 {
    font-size: 2.2rem !important;
}

h2 {
    font-size: 1.8rem !important;
}

h3 {
    font-size: 1.4rem !important;
}
</style>
''', unsafe_allow_html=True)


# Arcadis official brand color theme
ARCADIS_COLORS = {
    "primary": "#FF6900", # Arcadis Orange
    "secondary": "#4D4D4F", # Arcadis Dark Gray
    "background": "#FFFFFF", # White
    "light_gray": "#F5F5F5", # Light Gray for backgrounds
    "text": "#4D4D4F", # Dark Gray text
    "success": "#28A745", # Green
    "warning": "#FFC107", # Amber
    "error": "#DC3545" # Red
}

# Add Arcadis styling
def local_css():
    st.markdown("""
    <style>
        /* Complete Arcadis styling based on RiskLensPro */
        *, *::before, *::after,
        .stTabs [role="tablist"] [role="tab"],
        .stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"],
        div[data-testid="stVerticalBlock"], 
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stForm"],
        div[data-testid="stMetric"],
        div.stButton > button,
        div[data-testid="stImage"],
        select, input, textarea,
        .stMarkdown p, li, ul, ol,
        .st-bd, .st-bx, .st-by, .st-bz,
        .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj,
        .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv,
        span.css-10trblm, span.css-16idsys {  
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        }

        /* Modern theme colors - RiskLensPro inspired with Arcadis branded colors */
        :root {
            --primary-color: #FF6900;     /* Arcadis Orange */
            --secondary-color: #4D4D4F;   /* Arcadis Dark Gray */
            --accent-color: #2C3E50;      /* Dark Blue-Gray */
            --background-color: #F8F9FA;  /* Light Gray Background */
            --card-bg-color: #FFFFFF;     /* White for cards */
            --sidebar-bg: #2C3E50;        /* Dark Sidebar */
            --text-color: #333333;        /* Dark Text */
            --text-light: #FFFFFF;        /* Light Text */
            --success-color: #28A745;     /* Green */
            --warning-color: #FFC107;     /* Amber */
            --danger-color: #DC3545;      /* Red */
            --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
            --hover-transition: all 0.3s ease;
        }

        /* Modern app styling with dark sidebar and light content area */
        .stApp {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Main area styling */
        .main .block-container {
            padding: 1rem 2rem 2rem 2rem !important;
            max-width: 100% !important;
        }
        
        /* Sidebar - Dark styling */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: none !important;
        }
        
        /* All content in sidebar should be white */
        [data-testid="stSidebar"] * {
            color: var(--text-light) !important;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4 {
            color: var(--text-light) !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown a {
            color: var(--primary-color) !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--accent-color) !important;
            font-weight: 600 !important;
        }
        
        h1 {
            font-size: 2.2rem !important;
            margin-bottom: 1rem !important;
        }
        
        h2 {
            font-size: 1.8rem !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.8rem !important;
        }
        
        h3 {
            font-size: 1.4rem !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.6rem !important;
        }
        
        /* Regular text */
        p, li, div, span {
            color: var(--text-color) !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
        }
        
        /* Buttons - Modern styling */
        .stButton > button, div.stButton > button:first-child {
            background-color: var(--primary-color) !important;
            color: white !important;
            border: none !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-size: 0.9rem !important;
            transition: var(--hover-transition) !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15) !important;
        }
        
        .stButton > button:hover, div.stButton > button:hover {
            background-color: #E65800 !important; /* Darker orange on hover */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Tab styling - More aggressive styling to ensure visibility */
        div[data-testid="stTabs"] {
            border-bottom: 1px solid #e0e0e0 !important;
            margin-bottom: 2rem !important;
        }
        
        div[data-testid="stTabs"] > div[role="tablist"] {
            background-color: transparent !important;
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
            margin-bottom: 0 !important;
        }
        
        /* Override any Streamlit default styling for tabs with !important */
        div[data-testid="stTabs"] [role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"],
        div[data-testid="stTabs"] button {
            background-color: #f0f0f0 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 1rem 1.5rem !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            color: #333333 !important;
            border: 1px solid #e0e0e0 !important;
            border-bottom: none !important;
            margin-right: 0.5rem !important;
            position: relative !important;
            bottom: -1px !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        /* Make sure the text inside tabs is visible */
        div[data-testid="stTabs"] [role="tab"] div,
        div[data-testid="stTabs"] [role="tab"] p,
        div[data-testid="stTabs"] [role="tab"] span {
            color: #333333 !important;
            font-weight: 500 !important;
        }
        
        /* Explicitly style the active tab */
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #FF6900 !important; /* Arcadis Orange */
            color: white !important;
            font-weight: 600 !important;
            border: 1px solid #FF6900 !important;
            border-bottom: none !important;
            box-shadow: 0 -4px 10px rgba(0,0,0,0.05) !important;
        }
        
        /* Make sure text in active tab is white */
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] div,
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] span,
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] div,
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p,
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] span {
            color: white !important;
            font-weight: 600 !important;
        }
        
        /* Add hover effect to tabs */
        div[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]),
        div[data-testid="stTabs"] [data-baseweb="tab"]:hover:not([aria-selected="true"]),
        div[data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
            background-color: #e6e6e6 !important;
            transform: translateY(-2px) !important;
        }
        
        /* Card styling - Modern elevated cards */
        .element-container .stMarkdown div[data-testid="stMarkdownContainer"] > div {
            background-color: var(--card-bg-color) !important;
            border-radius: 10px !important;
            box-shadow: var(--card-shadow) !important;
            padding: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            transition: var(--hover-transition) !important;
        }
        
        /* Custom card classes */
        .card-container {
            background-color: var(--card-bg-color) !important;
            border-radius: 10px !important;
            box-shadow: var(--card-shadow) !important;
            padding: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            border-top: 4px solid var(--primary-color) !important;
        }
        
        /* KPI Cards */
        .kpi-card {
            background-color: var(--card-bg-color) !important;
            padding: 1.25rem !important;
            border-radius: 10px !important;
            box-shadow: var(--card-shadow) !important;
            margin-bottom: 1.5rem !important;
            border-left: 5px solid var(--primary-color) !important;
            transition: var(--hover-transition) !important;
            overflow: hidden !important;
            max-width: 100% !important;
            word-wrap: break-word !important;
        }
        
        .kpi-card:hover {
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.06) !important;
            transform: translateY(-3px) !important;
        }
        
        .kpi-card h3 {
            font-size: 1.1rem !important;
            margin-top: 0 !important;
            margin-bottom: 1rem !important;
            color: var(--accent-color) !important;
        }
        
        .kpi-card .value {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            margin: 0.5rem 0 !important;
            color: var(--primary-color) !important;
        }
        
        /* Status indicators */
        .status {
            font-size: 0.85rem !important;
            padding: 0.3rem 0.6rem !important;
            border-radius: 50px !important;
            display: inline-block !important;
            font-weight: 500 !important;
        }
        
        .status-green {
            background-color: rgba(40, 167, 69, 0.15) !important;
            color: var(--success-color) !important;
        }
        
        .status-amber {
            background-color: rgba(255, 193, 7, 0.15) !important;
            color: var(--warning-color) !important;
        }
        
        .status-red {
            background-color: rgba(220, 53, 69, 0.15) !important;
            color: var(--danger-color) !important;
        }
        
        /* Alert boxes */
        div[data-testid="stAlert"] {
            background-color: rgba(250, 250, 250, 0.9) !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            box-shadow: var(--card-shadow) !important;
            padding: 0.75rem 1rem !important;
        }
        
        div[data-testid="stAlert"] p {
            color: var(--text-color) !important;
            margin: 0 !important;
        }
        
        /* Form elements */
        div[data-baseweb="select"], 
        div[data-baseweb="input"], 
        div[data-baseweb="textarea"] {
            background-color: white !important;
            border-radius: 6px !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        div[data-baseweb="select"]:focus-within, 
        div[data-baseweb="input"]:focus-within, 
        div[data-baseweb="textarea"]:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(255, 105, 0, 0.2) !important;
        }
        
        /* Override for select boxes text */
        div[data-baseweb="select"] div {
            color: var(--text-color) !important;
        }
        
        /* Override for labels */
        label {
            color: var(--accent-color) !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        
        /* Progress bars */
        div[role="progressbar"] > div:first-child {
            background-color: rgba(0, 0, 0, 0.05) !important;
        }
        
        div[role="progressbar"] > div:nth-child(2) > div {
            background-color: var(--primary-color) !important;
        }
        
        /* Metrics */
        div[data-testid="stMetric"] {
            background-color: var(--card-bg-color) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            box-shadow: var(--card-shadow) !important;
        }
        
        div[data-testid="stMetric"] label {
            color: var(--accent-color) !important;
            font-weight: 600 !important;
        }
        
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: var(--primary-color) !important;
            font-weight: 700 !important;
        }
        
        /* Tables */
        div[data-testid="stTable"] {
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: var(--card-shadow) !important;
        }
        
        div[data-testid="stTable"] th {
            background-color: var(--accent-color) !important;
            color: white !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 0.75rem 1rem !important;
        }
        
        div[data-testid="stTable"] td {
            background-color: white !important;
            color: var(--text-color) !important;
            padding: 0.75rem 1rem !important;
            border-bottom: 1px solid #f0f0f0 !important;
        }
        
        /* Make sure all elements can be seen */
        * {
            color-scheme: light !important;
        }
    </style>
    """, unsafe_allow_html=True)
# Page config already set above

# Application constants and configurations
API_URL = "http://localhost:8000" if os.environ.get("API_URL") is None else os.environ.get("API_URL")

# Initialize session state for tracking application state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = None
    
if "client_name" not in st.session_state:
    st.session_state.client_name = "Thames Water"
    
if "client_theme" not in st.session_state:
    st.session_state.client_theme = {
        "primary_color": "#005670",
        "secondary_color": "#00A1D6",
        "success_color": "#28A745",
        "warning_color": "#FFB107",
        "danger_color": "#FF4B4B",
        "logo_url": None
    }

# Authentication management
def login_page():
    # Apply Arcadis styling
    local_css()
    
    # Header with Arcadis official branding
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <img src="https://raw.githubusercontent.com/arcadis-logo/branding/main/arcadis-logo.png" width="250" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMzAiPg0KICAgIDxwYXRoIGZpbGw9IiNGRjY5MDAiIGQ9Ik0yMC40LDcuMWMtMS4yLDAtMi4yLDAuMy0zLjIsMC44Yy0xLDAuNS0xLjksMS4zLTIuNywyLjFjLTAuOCwwLjktMS41LDEuOS0yLDMuMWMtMC41LDEuMi0wLjgsMi41LTAuOCwzLjljMCwxLjQsMC4zLDIuNywwLjgsMy45YzAuNSwxLjIsMS4yLDIuMiwyLDMuMWMwLjgsMC45LDEuNywxLjYsMi43LDIuMWMxLDAuNSwyLDAuOCwzLjIsMC44YzEuMiwwLDIuMi0wLjMsMy4yLTAuOGMxLTAuNSwxLjktMS4zLDIuNy0yLjFjMC44LTAuOSwxLjUtMS45LDItMy4xYzAuNS0xLjIsMC44LTIuNSwwLjgtMy45YzAtMS40LTAuMy0yLjctMC44LTMuOWMtMC41LTEuMi0xLjItMi4yLTItMy4xYy0wLjgtMC45LTEuNy0xLjYtMi43LTIuMUMyMi43LDcuNCwyMS41LDcuMSwyMC40LDcuMXoiLz4NCiAgICA8cGF0aCBmaWxsPSIjNEQ0RDRGIiBkPSJNMzUuNCwxNC4yVjE5aDMuOXYxLjdoLTUuOHYtMTBoOS40djEuN2gtNy41djEuOEgzOS45djEuN0gzNS40ek00My44LDEwLjZoMi4xbDQuMSw2LjV2LTYuNWgxLjl2MTBoLTJsLTQuMS02LjV2Ni41aC0xLjlWMTAuNnpNNTcuNiwxOC41YzAsMS43LTEuNCwzLjEtMy4xLDMuMWMtMS43LDAtMy4xLTEuNC0zLjEtMy4xdi03LjloMS45djcuOWMwLDAuNiwwLjUsMS4yLDEuMiwxLjJjMC42LDAsMS4yLTAuNSwxLjItMS4ydi03LjloMS45VjE4LjV6Ek01OC44LDEwLjZoNS41YzEuNCwwLDIuNiwxLjEsMi42LDIuNmMwLDEuMS0wLjcsMS45LTEuNywyLjNjMC45LDAuNCwxLjQsMS4yLDEuNSwyLjRsMC4xLDIuOGgtMmwtMC4xLTIuNGMwLTEuMy0wLjQtMS44LTEuNy0xLjhoLTIuM3Y0LjJoLTEuOVYxMC42eiBNNjAuNywxNC44aDIuOWMwLjYsMCwxLjItMC41LDEuMi0xLjJjMC0wLjYtMC41LTEuMi0xLjItMS4yaC0yLjlWMTQuOHpNNjkuNCwxMC42aDEuOXY4LjNoNS4zVjIwLjZoLTcuMlYxMC42ek03OC41LDEwLjZoMS45djEwLjFoLTEuOVYxMC42ek04Mi4yLDEwLjZoMi4ybDUsMTBoLTIuMWwtMS4xLTIuMmgtNS42bC0xLjEsMi4yaC0yLjJMODIuMiwxMC42eiBNODIuMSwxNi43bC0xLjctMy41bC0xLjcsMy41SDgyLjF6Ij48L3BhdGg+DQo8L3N2Zz4=';">
        <h1 style="color: #4D4D4F; margin-top: 20px;">Arcadis Analytics Suite</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #FF6900;'>Demo Mode Active</h3>", unsafe_allow_html=True)
    st.info("Authentication is currently bypassed for demonstration purposes. Select an organization and click the button below to access the dashboard.")
    
    # Demo selection for different organizations with Arcadis styling
    tenant = st.selectbox("Select Organization for Demo", 
                         ["Thames Water", "Southern Water", "Anglian Water", "Yorkshire Water", "Arcadis Water"])
    
    if st.button("Access Dashboard", key="login_btn"):
        # Auto-login with demo credentials
        st.session_state.authenticated = True
        st.session_state.username = "demo_user"
        st.session_state.tenant_id = "demo_tenant"
        st.session_state.client_name = tenant
        st.rerun()
    
    st.markdown("---")
    
    # Two column layout for platform description
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #4D4D4F;'>About the Platform</h3>", unsafe_allow_html=True)
        st.markdown("""
        Arcadis Analytics Suite provides organizations with:
        - **Secure integration** with operational data systems
        - **Advanced analytics** and visualization
        - **Financial impact modeling** and ROI calculation
        - **Regulatory reporting** and compliance tracking
        - **Prescriptive recommendations** and actionable insights
        """)
    
    with col2:
        # Modern analytics visualization
        st.markdown("""
        <div style="background-color:#F8F9FA; height:200px; display:flex; align-items:center; justify-content:center; border-radius:10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); overflow:hidden; position:relative;">
            <div style="position:absolute; top:0; right:0; width:120px; height:120px; background-color:#FF6900; opacity:0.1; border-radius:0 0 0 120px;"></div>
            <div style="position:absolute; bottom:0; left:0; width:80px; height:80px; background-color:#2C3E50; opacity:0.1; border-radius:0 80px 0 0;"></div>
            <div style="text-align: center; position:relative; z-index:2;">
                <span style="font-size:3.5rem; color:#FF6900; display:block; margin-bottom:15px;">📊</span>
                <p style="color:#2C3E50; margin-top: 10px; font-size:1.2rem; font-weight:500;">Interactive Analytics Suite</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Platform benefits
    st.markdown("<h3 style='color: #4D4D4F; margin-top: 30px;'>Key Features</h3>", unsafe_allow_html=True)
    
    # Features in cards
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div style="background-color:white; padding:24px; border-radius:10px; min-height:180px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1); border-top: 4px solid #FF6900; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-5px)';this.style.boxShadow='0 10px 20px rgba(0, 0, 0, 0.1), 0 4px 8px rgba(0, 0, 0, 0.06)';" onmouseout="this.style.transform='translateY(0px)';this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1)'">
            <h4 style="color:#2C3E50; margin-top:0; font-size:18px; font-weight:600;">Data Integration</h4>
            <ul style="margin-bottom:0; color:#333333; padding-left:20px;">
                <li style="margin-bottom:8px;">Multi-tenant architecture</li>
                <li style="margin-bottom:8px;">White-labeling capabilities</li>
                <li style="margin-bottom:8px;">Secure data validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with feature_col2:
        st.markdown("""
        <div style="background-color:white; padding:24px; border-radius:10px; min-height:180px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1); border-top: 4px solid #FF6900; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-5px)';this.style.boxShadow='0 10px 20px rgba(0, 0, 0, 0.1), 0 4px 8px rgba(0, 0, 0, 0.06)';" onmouseout="this.style.transform='translateY(0px)';this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1)'">
            <h4 style="color:#2C3E50; margin-top:0; font-size:18px; font-weight:600;">Smart Analytics</h4>
            <ul style="margin-bottom:0; color:#333333; padding-left:20px;">
                <li style="margin-bottom:8px;">Predictive modeling</li>
                <li style="margin-bottom:8px;">Automated insights</li>
                <li style="margin-bottom:8px;">Anomaly detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with feature_col3:
        st.markdown("""
        <div style="background-color:white; padding:24px; border-radius:10px; min-height:180px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1); border-top: 4px solid #FF6900; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-5px)';this.style.boxShadow='0 10px 20px rgba(0, 0, 0, 0.1), 0 4px 8px rgba(0, 0, 0, 0.06)';" onmouseout="this.style.transform='translateY(0px)';this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1)'">
            <h4 style="color:#2C3E50; margin-top:0; font-size:18px; font-weight:600;">Business Impact</h4>
            <ul style="margin-bottom:0; color:#333333; padding-left:20px;">
                <li style="margin-bottom:8px;">Financial impact calculator</li>
                <li style="margin-bottom:8px;">Regulatory compliance tracking</li>
                <li style="margin-bottom:8px;">Executive dashboards</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    # Call to action with Arcadis colors
    st.markdown("""
    <div style="text-align: center; padding: 30px 30px; background-color: #f8f9fa; border-radius: 12px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);">
        <p style="font-size: 1.4rem; color: #2C3E50; font-weight: 600; margin-bottom: 12px;">Ready to explore powerful analytics?</p>
        <p style="color: #4D4D4F; font-size: 1.1rem; margin-bottom: 20px;">Select your organization above and click "Access Dashboard" to begin.</p>
        <div style="width: 100px; height: 4px; background-color: #FF6900; margin: 0 auto;"></div>
    </div>
    """, unsafe_allow_html=True)

# Main application function
def main():
    # Apply direct styling fixes from app_theme
    fix_all_styling()
    
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Apply client theme
    # Convert client name to a format used by the whitelist component
    client_id = st.session_state.client_name.lower().replace(' ', '_')
    apply_client_theme(client_id)
    
    # Header with client name and user info
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.client_theme.get("logo_url"):
            st.image(st.session_state.client_theme.get("logo_url"), width=200)
        else:
            st.title(st.session_state.client_name + " Analytics Suite")
    with col2:
        st.markdown(f"Welcome, **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.tenant_id = None
            st.rerun()
    
    # Main tab navigation with clearly visible tab names and force tab styling inline
    st.markdown('''
    <style>
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button[aria-selected="true"] {
        background-color: #FF6900 !important;
        color: white !important;
    }
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button {
        background-color: #f0f0f0 !important;
        color: #333333 !important;
        border-radius: 8px 8px 0 0 !important;
    }
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button p {
        color: inherit !important;
    }
    </style>
    ''', unsafe_allow_html=True)
    
    tabs = st.tabs(["📋 Welcome", "📊 Dashboard", "✓ Data Quality", "💡 Insights", "💰 Financial Impact", "📑 Reports"])
    
    # Apply the most aggressive tab styling possible right after creating tabs
    st.markdown("""
    <style>
    /* Force tab styling - final override */
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] {
        background-color: transparent !important;
    }
    
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button {
        background-color: #f0f0f0 !important;
        color: #333333 !important;
        border-radius: 5px 5px 0 0 !important;
        padding: 10px 16px !important;
        margin-right: 4px !important;
        border: 1px solid #e0e0e0 !important;
        border-bottom: none !important;
    }
    
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button[aria-selected="true"] {
        background-color: #FF6900 !important;
        color: white !important;
        font-weight: 600 !important;
        border: 1px solid #FF6900 !important;
    }
    
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button p,
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button div {
        color: #333333 !important;
    }
    
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button[aria-selected="true"] p,
    div[data-testid="stTabs"] > div > div > div > div[role="tablist"] button[aria-selected="true"] div {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for data upload only
    with st.sidebar:
        # Arcadis logo at the top (official logo)
        st.markdown("""
        <div style="text-align: center; padding-bottom: 20px;">
            <img src="https://raw.githubusercontent.com/arcadis-logo/branding/main/arcadis-logo.png" width="200" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMzAiPg0KICAgIDxwYXRoIGZpbGw9IiNGRjY5MDAiIGQ9Ik0yMC40LDcuMWMtMS4yLDAtMi4yLDAuMy0zLjIsMC44Yy0xLDAuNS0xLjksMS4zLTIuNywyLjFjLTAuOCwwLjktMS41LDEuOS0yLDMuMWMtMC41LDEuMi0wLjgsMi41LTAuOCwzLjljMCwxLjQsMC4zLDIuNywwLjgsMy45YzAuNSwxLjIsMS4yLDIuMiwyLDMuMWMwLjgsMC45LDEuNywxLjYsMi43LDIuMWMxLDAuNSwyLDAuOCwzLjIsMC44YzEuMiwwLDIuMi0wLjMsMy4yLTAuOGMxLTAuNSwxLjktMS4zLDIuNy0yLjFjMC44LTAuOSwxLjUtMS45LDItMy4xYzAuNS0xLjIsMC44LTIuNSwwLjgtMy45YzAtMS40LTAuMy0yLjctMC44LTMuOWMtMC41LTEuMi0xLjItMi4yLTItMy4xYy0wLjgtMC45LTEuNy0xLjYtMi43LTIuMUMyMi43LDcuNCwyMS41LDcuMSwyMC40LDcuMXoiLz4NCiAgICA8cGF0aCBmaWxsPSIjNEQ0RDRGIiBkPSJNMzUuNCwxNC4yVjE5aDMuOXYxLjdoLTUuOHYtMTBoOS40djEuN2gtNy41djEuOEgzOS45djEuN0gzNS40ek00My44LDEwLjZoMi4xbDQuMSw2LjV2LTYuNWgxLjl2MTBoLTJsLTQuMS02LjV2Ni41aC0xLjlWMTAuNnpNNTcuNiwxOC41YzAsMS43LTEuNCwzLjEtMy4xLDMuMWMtMS43LDAtMy4xLTEuNC0zLjEtMy4xdi03LjloMS45djcuOWMwLDAuNiwwLjUsMS4yLDEuMiwxLjJjMC42LDAsMS4yLTAuNSwxLjItMS4ydi03LjloMS45VjE4LjV6Ik01OC44LDEwLjZoNS41YzEuNCwwLDIuNiwxLjEsMi42LDIuNmMwLDEuMS0wLjcsMS45LTEuNywyLjNjMC45LDAuNCwxLjQsMS4yLDEuNSwyLjRsMC4xLDIuOGgtMmwtMC4xLTIuNGMwLTEuMy0wLjQtMS44LTEuNy0xLjhoLTIuM3Y0LjJoLTEuOVYxMC42eiBNNjAuNywxNC44aDIuOWMwLjYsMCwxLjItMC41LDEuMi0xLjJjMC0wLjYtMC41LTEuMi0xLjItMS4yaC0yLjlWMTQuOHpNNjkuNCwxMC42aDEuOXY4LjNoNS4zVjIwLjZoLTcuMlYxMC42ek03OC41LDEwLjZoMS45djEwLjFoLTEuOVYxMC42ek04Mi4yLDEwLjZoMi4ybDUsMTBoLTIuMWwtMS4xLTIuMmgtNS42bC0xLjEsMi4yaC0yLjJMODIuMiwxMC42eiBNODIuMSwxNi43bC0xLjctMy41bC0xLjcsMy41SDgyLjF6Ij48L3BhdGg+DQo8L3N2Zz4=';">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='color: white;'>Data Upload</h2>", unsafe_allow_html=True)
        
        # Add Demo Data button at the top for quick exploration
        st.markdown("<h3 style='color: #FF6900; opacity: 0.9;'>Quick Start</h3>", unsafe_allow_html=True)
        demo_data_col1, demo_data_col2 = st.columns(2)
        with demo_data_col1:
            if st.button("📊 Load Demo Data", use_container_width=True, key="load_demo_btn"):
                st.session_state.demo_data_loaded = True
                st.success("Demo data loaded successfully!")
                st.info("6 datasets loaded: Consumption, Meters, Treatment Plants, Infrastructure, Customer, and Financial data")
        with demo_data_col2:
            if st.button("🧹 Clear Data", use_container_width=True, key="clear_data_btn"):
                if "demo_data_loaded" in st.session_state:
                    del st.session_state.demo_data_loaded
                st.info("Data cleared. Upload your own data or load demo data again.")
        
        st.markdown("---")
        st.markdown("### Upload Your Data")
        
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        dataset_type = st.selectbox(
            "Dataset Type", 
            ["Water Consumption", "Meter Readings", "Treatment Plant Data", "Infrastructure Status", "Customer Data", "Financial Data"]
        )
        
        if uploaded_file is not None:
            st.success(f"File {uploaded_file.name} uploaded successfully")
            st.info("Identified as " + dataset_type + " data. The system will intelligently process and join this with existing datasets.")
            
            # Process button for uploaded data
            if st.button("Process Uploaded Data"):
                with st.spinner("Processing data and generating insights..."):
                    # Simulate processing delay
                    time.sleep(2)
                    st.success("Data processed successfully! Insights have been generated.")
                    st.session_state.user_data_loaded = True
        
        st.markdown("---")
        st.markdown("### Data Management")
        st.caption("Manage and view your uploaded datasets")
        
        # Show data status
        if "demo_data_loaded" in st.session_state or "user_data_loaded" in st.session_state or uploaded_file is not None:
            st.markdown("**Loaded Datasets:**")
            if "demo_data_loaded" in st.session_state:
                st.markdown("✅ Demo Water Consumption Data")
                st.markdown("✅ Demo Meter Readings")
                st.markdown("✅ Demo Treatment Plant Data")
                st.markdown("✅ Demo Infrastructure Data")
                st.markdown("✅ Demo Customer Data")
                st.markdown("✅ Demo Financial Data")
            if "user_data_loaded" in st.session_state or uploaded_file is not None:
                st.markdown(f"✅ User {dataset_type} Data")
        else:
            st.warning("No datasets loaded. Please upload data or load demo data.")
        
        # Data management buttons
        if st.button("View Datasets"):
            # This would show a modal or navigate to a data catalog view
            st.info("This would show all datasets currently in the system with metadata and preview options.")
            
        if st.button("Data Dictionary"):
            # This would show metadata about expected columns/format
            st.info("This would show the expected format and columns for each dataset type with descriptions.")
            
        # Add data quality overview
        if "demo_data_loaded" in st.session_state or "user_data_loaded" in st.session_state:
            st.markdown("---")
            st.markdown("### Data Quality Summary")
            data_quality_score = 92 if "demo_data_loaded" in st.session_state else 78
            st.progress(data_quality_score/100, text=f"Overall Quality: {data_quality_score}%")
            
            # Quick quality metrics
            quality_col1, quality_col2 = st.columns(2)
            with quality_col1:
                st.markdown("<div style='color: rgba(255,255,255,0.8);'>Completeness</div>", unsafe_allow_html=True)
                st.progress(0.95, text="95%")
            with quality_col2:
                st.markdown("<div style='color: rgba(255,255,255,0.8);'>Accuracy</div>", unsafe_allow_html=True)
                st.progress(0.89, text="89%")

    
    # Content for each tab
    # 1. Welcome tab
    with tabs[0]:
        st.markdown("<h2 style='color: #000000; font-weight: 700;'>Welcome to Arcadis Analytics Suite</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #000000; font-size: 16px; font-weight: 400; line-height: 1.6;">Arcadis Analytics Suite is designed to help organizations gain powerful insights from their operational data,
        improve efficiency, reduce costs, and ensure regulatory compliance.</p>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #000000; font-weight: 600; margin-top: 20px;'>How to Use This Platform</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <ol style="color: #000000; font-size: 15px; font-weight: 400; line-height: 1.6; margin-left: 25px;">
            <li><strong>Upload Your Data</strong>: Use the sidebar to upload your operational data files. The system recognizes different data types and automatically processes them.</li>
            <li><strong>Explore the Dashboard</strong>: View key performance indicators (KPIs) and trends for your water utility operations.</li>
            <li><strong>Data Quality</strong>: Analyze the quality and completeness of your data to ensure reliable insights.</li>
            <li><strong>Insights</strong>: Discover patterns, anomalies, and actionable recommendations based on your data.</li>
            <li><strong>Financial Impact</strong>: Quantify the financial impact of operational changes and improvements.</li>
            <li><strong>Reports</strong>: Generate and export compliance and operational reports for stakeholders.</li>
        </ol>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #000000; font-weight: 600; margin-top: 20px;'>Key Features</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <ul style="color: #000000; font-size: 15px; font-weight: 400; line-height: 1.6; margin-left: 25px;">
            <li><strong>Intelligent Data Processing</strong>: Upload your data once, and our system automatically processes, validates, and joins related datasets.</li>
            <li><strong>Interactive Visualizations</strong>: Explore your data through intuitive charts and graphs.</li>
            <li><strong>Anomaly Detection</strong>: Automatically identify unusual patterns that may indicate issues.</li>
            <li><strong>Financial Modeling</strong>: Calculate ROI for operational improvements.</li>
            <li><strong>Trend Analysis</strong>: Track performance metrics over time and forecast future trends.</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #000000; font-weight: 600; margin-top: 20px;'>Getting Started</h3>", unsafe_allow_html=True)
        
        # Quick start actions in two columns
        quick_col1, quick_col2 = st.columns(2)
        with quick_col1:
            st.markdown("""
            <div style='background-color: #f7f7f7; padding: 15px; border-radius: 5px; border-left: 4px solid #009FDA; color: #000000;'>
                <p style='margin: 0; font-weight: 600; font-size: 16px;'>Option 1: Use Demo Data</p>
                <p style='margin-top: 10px; color: #000000;'>Click the <strong>📊 Load Demo Data</strong> button in the sidebar to instantly load pre-processed water utility datasets and explore the platform's capabilities.</p>
            </div>
            """, unsafe_allow_html=True)
        with quick_col2:
            st.markdown("""
            <div style='background-color: #f7f7f7; padding: 15px; border-radius: 5px; border-left: 4px solid #FF6900; color: #000000;'>
                <p style='margin: 0; font-weight: 600; font-size: 16px;'>Option 2: Upload Your Data</p>
                <p style='margin-top: 10px; color: #000000;'>Use the file uploader in the sidebar to upload your own CSV or Excel files. The system will automatically process and analyze your data.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Demo data benefits
        st.markdown("<h3 style='color: #000000; font-weight: 600; margin-top: 20px;'>Demo Data Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #f0fff4; padding: 15px 20px; border-radius: 5px; border-left: 4px solid #28a745; color: #000000;">
            <p style="color: #000000; font-weight: 400;">The demo data includes:</p>
            <ul style="color: #000000; list-style-type: disc; padding-left: 25px;">
                <li><strong>3 years</strong> of historical water consumption data</li>
                <li><strong>5,000+ meter readings</strong> with geospatial information</li>
                <li><strong>Treatment plant operational data</strong> from 4 facilities</li>
                <li><strong>Infrastructure status reports</strong> for pipelines and pumps</li>
                <li><strong>Customer billing and service records</strong></li>
                <li><strong>Financial performance metrics</strong></li>
            </ul>
            <p style="color: #000000; margin-top: 10px;">All datasets are pre-joined and cleaned for immediate analysis. Simply click the '<strong>📊 Load Demo Data</strong>' button in the sidebar to begin exploring.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Added prominent call to action with official Arcadis colors
        demo_cta_col1, demo_cta_col2, demo_cta_col3 = st.columns([1,2,1])
        with demo_cta_col2:
            st.markdown("""
            <div style="text-align: center; padding: 20px; margin: 20px 0; background-color: #F5F5F5; border-radius: 10px; border: 1px solid #FF6900;">
                <h3 style="color: #FF6900; font-weight: 600;">Ready to explore?</h3>
                <p style="color: #000000; font-size: 15px;">Click the <strong>📊 Load Demo Data</strong> button in the sidebar to instantly populate all dashboards with sample water utility data.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 2. Dashboard tab
    with tabs[1]:
        render_dashboard()
    
    # 3. Data Quality tab
    with tabs[2]:
        st.markdown("## Data Quality Analysis")
        # Using some components from the data_quality module we've created
        try:
            st.markdown("### Data Quality Overview")
            st.markdown("This section shows the quality of your uploaded and connected datasets, highlighting any issues that may affect analysis.")
            
            # Sample datasets for selection - in real use, this would be populated from actual data
            dataset = st.selectbox(
                "Select Dataset", 
                ["Water Consumption Data", "Treatment Plant Operations", "Customer Billing Records", "Infrastructure Status"]
            )
            
            # Sample quality dimensions
            quality_dimensions = st.multiselect(
                "Quality Dimensions to Analyze",
                ["Completeness", "Accuracy", "Consistency", "Timeliness", "Validity", "Uniqueness"],
                default=["Completeness", "Accuracy"]
            )
            
            if st.button("Run Quality Check"):
                with st.spinner("Analyzing data quality..."):
                    # Call our data quality module with proper parameters
                    quality_results = run_data_quality_check(
                        data=None,
                        tenant_id="default",
                        dataset_id=dataset,
                        check_types=quality_dimensions
                    )
                    
                    # Display the quality score
                    st.markdown(f"### Quality Score: {quality_results['overall_score']}/100")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Completeness", "92%", "+2%")
                    with col2:
                        st.metric("Accuracy", "87%", "-1%")
                    with col3:
                        st.metric("Consistency", "93%", "+4%")
                    
                    st.markdown("### Quality Issues Detected")
                    st.markdown("""                    
                    - **4% missing values** in water consumption readings (4,219 records)
                    - **2 outliers** detected in pressure readings that exceed 3 standard deviations
                    - **12 duplicate customer records** identified with minor variations
                    """)
                    
                    st.markdown("### Recommended Actions")
                    st.markdown("""
                    <div style="background-color: #f1f8ff; padding: 20px; border-radius: 5px; border-left: 5px solid #0366d6; color: #24292e; font-weight: normal;">
                    <ol style="margin-left: 15px; color: #24292e;">
                        <li><strong>Verify pressure sensor calibration</strong> at Northern Treatment Plant</li>
                        <li><strong>Implement data validation rules</strong> for customer record entry</li>
                        <li><strong>Review data collection process</strong> for consumption readings</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error in Data Quality module: {str(e)}")
    
    # 4. Insights tab
    with tabs[3]:
        st.markdown("## Data Insights")
        try:
            st.markdown("### Automated Insights")
            st.markdown("This section uses machine learning to identify patterns, trends, and anomalies in your data.")
            
            # Create more visible styling for radio buttons and text
            st.markdown("""
            <style>
            /* Overall style for the radio button container */
            div.row-widget.stRadio > div {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                background-color: #f7f7f7;
                padding: 15px 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid #e0e0e0;
            }
            
            /* Individual radio button styling */
            div.row-widget.stRadio > div[role="radiogroup"] > label {
                background-color: white;
                padding: 15px 20px;
                border-radius: 5px;
                margin-right: 10px;
                border: 2px solid #dcdcdc;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.2s ease;
                position: relative;
                display: flex;
                align-items: center;
            }
            
            /* Radio button label text - critical fix for visibility */
            div.row-widget.stRadio > div[role="radiogroup"] > label > div:last-child {
                color: #000000;
                font-weight: 600;
                font-size: 16px;
                margin-left: 8px;
                text-shadow: 0px 0px 1px rgba(0,0,0,0.1);
            }
            
            /* Hover state */
            div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
                background-color: #f8f8f8;
                border-color: #FF6900;
                box-shadow: 0 3px 6px rgba(0,0,0,0.15);
            }
            
            /* Radio button circles */
            div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
                background-color: #FF6900;
                border-color: #FF6900;
                transform: scale(1.2);
            }
            
            /* Selected radio button styling */
            div.row-widget.stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
                background-color: #fff8f5;
                border-color: #FF6900;
                box-shadow: 0 3px 6px rgba(255,105,0,0.2);
            }
            
            div.row-widget.stRadio > div[role="radiogroup"] > label[aria-checked="true"] > div:last-child {
                color: #FF6900;
                font-weight: 700;
            }
            
            /* Insight Type header styling */
            [data-testid="stVerticalBlock"] > div:has(div.row-widget.stRadio) > div:first-child {
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 15px;
                color: #000000;
                text-shadow: 0px 0px 1px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            insight_type = st.radio(
                "Insight Type",
                options=["Anomaly Detection", "Trend Analysis", "Correlation Analysis", "Predictive Insights"],
                horizontal=True
            )
            
            # Demo content based on selected insight type
            if insight_type == "Anomaly Detection":
                st.subheader("Anomalies Detected")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div style="background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); color: #24292e;">
                        <ul style="list-style-type: none; padding-left: 5px; margin-bottom: 0;">
                            <li style="margin-bottom: 10px;">⚠️ <strong style="color: #dc3545;">Unusual water consumption</strong> in Northern District on April 28th (67% above normal)</li>
                            <li style="margin-bottom: 10px;">⚠️ <strong style="color: #dc3545;">Pressure drop</strong> detected at Main Pump Station between 2-4 AM on May 1st</li>
                            <li style="margin-bottom: 0px;">⚠️ <strong style="color: #e36209;">3 billing anomalies</strong> identified with consumption-to-billing ratio outside expected range</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    # Placeholder for anomaly chart with improved visibility
                    st.markdown("""
                    <div style="background-color:#ffffff; height:230px; border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 15px; position: relative; border: 1px solid #e0e0e0;">
                        <div style="position: absolute; top: 10px; left: 15px; font-weight: 600; color: #24292e; font-size: 14px;">Anomaly Detection Chart</div>
                        <div style="display: flex; align-items:center; justify-content:center; height: 100%; width: 100%;">
                            <div style="text-align: center;">
                                <div style="margin-bottom: 15px;">
                                    <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M21 21H4.6C4.03995 21 3.75992 21 3.54601 20.891C3.35785 20.7951 3.20487 20.6422 3.10899 20.454C3 20.2401 3 19.9601 3 19.4V3" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        <path d="M7 15L11 10L15 13L20 7" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        <circle cx="16" cy="16" r="6" fill="#FFECE3" stroke="#FF6900" stroke-width="2"/>
                                        <path d="M16 13V16L18 18" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    </svg>
                                </div>
                                <span style="color: #6c757d; font-size: 0.95rem;">Chart appears when data is loaded</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif insight_type == "Trend Analysis":
                st.subheader("Key Trends")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div style="background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); color: #24292e;">
                        <ul style="list-style-type: none; padding-left: 5px; margin-bottom: 0;">
                            <li style="margin-bottom: 10px;">📈 <strong style="color: #218838;">Increasing water consumption</strong> trend in Residential Sector (5.2% year-over-year)</li>
                            <li style="margin-bottom: 10px;">📉 <strong style="color: #dc3545;">Decreasing pressure levels</strong> in Western District (1.2% month-over-month)</li>
                            <li style="margin-bottom: 0px;">📊 <strong style="color: #218838;">Improving treatment efficiency</strong> at Northern Plant (3.7% increase in last quarter)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    # Placeholder for trend chart with improved visibility
                    st.markdown("""
                    <div style="background-color:#ffffff; height:230px; border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 15px; position: relative; border: 1px solid #e0e0e0;">
                        <div style="position: absolute; top: 10px; left: 15px; font-weight: 600; color: #24292e; font-size: 14px;">Trend Analysis Chart</div>
                        <div style="display: flex; align-items:center; justify-content:center; height: 100%; width: 100%;">
                            <div style="text-align: center;">
                                <div style="margin-bottom: 15px;">
                                    <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M3 3V21H21" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        <path d="M3 17L9 11L13 15L21 7" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        <path d="M21 12V7H16" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    </svg>
                                </div>
                                <span style="color: #6c757d; font-size: 0.95rem;">Chart appears when data is loaded</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif insight_type == "Correlation Analysis":
                st.subheader("Significant Correlations")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div style="background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); color: #24292e;">
                        <ul style="list-style-type: none; padding-left: 5px; margin-bottom: 0;">
                            <li style="margin-bottom: 10px;"><span style="font-weight: bold; color: #0366d6;">Strong correlation (0.87)</span> between rainfall and treatment plant inflow</li>
                            <li style="margin-bottom: 10px;"><span style="font-weight: bold; color: #d73a49;">Negative correlation (-0.76)</span> between temperature and water consumption</li>
                            <li style="margin-bottom: 0px;"><span style="font-weight: bold; color: #e36209;">Moderate correlation (0.62)</span> between system pressure and leak incidents</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    # Placeholder for correlation matrix with improved visibility
                    st.markdown("""
                    <div style="background-color:#ffffff; height:230px; border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 15px; position: relative; border: 1px solid #e0e0e0;">
                        <div style="position: absolute; top: 10px; left: 15px; font-weight: 600; color: #24292e; font-size: 14px;">Correlation Matrix</div>
                        <div style="display: flex; align-items:center; justify-content:center; height: 100%; width: 100%;">
                            <div style="text-align: center;">
                                <div style="margin-bottom: 15px;">
                                    <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <rect x="3" y="3" width="7" height="7" rx="1" stroke="#FF6900" stroke-width="2"/>
                                        <rect x="14" y="3" width="7" height="7" rx="1" stroke="#FF6900" stroke-width="2"/>
                                        <rect x="14" y="14" width="7" height="7" rx="1" stroke="#FF6900" stroke-width="2"/>
                                        <rect x="3" y="14" width="7" height="7" rx="1" stroke="#FF6900" stroke-width="2"/>
                                        <path d="M6.5 7L6.5 17" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 3"/>
                                        <path d="M17.5 7L17.5 17" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 3"/>
                                        <path d="M7 6.5L17 6.5" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 3"/>
                                        <path d="M7 17.5L17 17.5" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 3"/>
                                    </svg>
                                </div>
                                <span style="color: #6c757d; font-size: 0.95rem;">Matrix appears when data is loaded</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:  # Predictive Insights
                st.subheader("Predictive Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div style="background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); color: #24292e;">
                        <ul style="list-style-type: none; padding-left: 5px; margin-bottom: 0;">
                            <li style="margin-bottom: 10px;">📈 <strong style="color: #218838;">Forecasted 8% increase</strong> in summer water demand based on weather predictions</li>
                            <li style="margin-bottom: 10px;">⚠️ <strong style="color: #dc3545;">Projected 12% probability</strong> of pressure issues in Sector B within 30 days</li>
                            <li style="margin-bottom: 0px;">📉 <strong style="color: #218838;">Expected 5% reduction</strong> in treatment chemicals needed next month</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    # Placeholder for forecast chart with improved visibility
                    st.markdown("""
                    <div style="background-color:#ffffff; height:230px; border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 15px; position: relative; border: 1px solid #e0e0e0;">
                        <div style="position: absolute; top: 10px; left: 15px; font-weight: 600; color: #24292e; font-size: 14px;">Forecast Chart</div>
                        <div style="display: flex; align-items:center; justify-content:center; height: 100%; width: 100%;">
                            <div style="text-align: center;">
                                <div style="margin-bottom: 15px;">
                                    <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M3 3V21H21" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        <path d="M7 10L11 14L16 9L20 13" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        <path d="M7 15L11 19" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.2 4"/>
                                        <path d="M16 14L20 18" stroke="#FF6900" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.2 4"/>
                                    </svg>
                                </div>
                                <span style="color: #6c757d; font-size: 0.95rem;">Chart appears when data is loaded</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### Action Recommendations")
            st.markdown("""
            <div style="background-color: #f1f8ff; padding: 20px; border-radius: 5px; border-left: 5px solid #0366d6; color: #24292e; font-weight: normal;">
            <ol style="margin-left: 15px; color: #24292e;">
                <li><span style="color: #d73a49; font-weight: bold;">High Priority:</span> Investigate unusual consumption in Northern District</li>
                <li><span style="color: #e36209; font-weight: bold;">Medium Priority:</span> Schedule preventative maintenance for Western District infrastructure</li>
                <li><span style="color: #2188ff; font-weight: bold;">Low Priority:</span> Review chemical ordering plan for potential cost savings</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in Insights module: {str(e)}")
    
    # 5. Financial Impact tab
    with tabs[4]:
        st.markdown("## Financial Impact Analysis")
        try:
            # Adding custom styling for Financial Impact tab headings and text
            st.markdown("""
            <style>
            /* Financial tab headings and text enhancements */
            [data-testid="stVerticalBlock"] > div > div:has(label:contains("Improvement Category")) {
                margin-top: 20px;
            }
            
            /* Slider and selectbox label style */
            div.row-widget.stSelectbox > div:first-child,
            div.row-widget.stSlider > div:first-child,
            div.row-widget.stNumberInput > div:first-child {
                color: #000000 !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                margin-bottom: 5px !important;
                text-shadow: 0px 0px 1px rgba(0,0,0,0.05) !important;
                background-color: #f8f8f8 !important;
                padding: 5px 10px !important;
                border-radius: 4px !important;
                border-left: 3px solid #FF6900 !important;
            }
            
            /* Improve slider track visibility */
            div.stSlider [data-baseweb="slider"] div {
                background-color: #e0e0e0 !important;
            }
            
            /* Slider handle styling */
            div.stSlider [data-baseweb="slider"] div[role="slider"] {
                background-color: #FF6900 !important;
                height: 1.6rem !important;
                width: 1.6rem !important;
                margin-top: -0.8rem !important;
                border-color: #FF6900 !important;
            }
            
            /* Number input styling */
            div.stNumberInput input {
                border-color: #ccc !important;
                color: #000000 !important;
                font-weight: 500 !important;
            }
            
            /* Focus states */
            div.stNumberInput input:focus {
                border-color: #FF6900 !important;
                box-shadow: 0 0 0 0.2rem rgba(255, 105, 0, 0.25) !important;
            }
            
            /* Make sure all text in the Financial tab is dark and visible */
            [data-testid="stVerticalBlock"] label,
            [data-testid="stVerticalBlock"] div.stMarkdown p {
                color: #000000 !important;
                font-weight: 500 !important;
            }
            
            /* Style headings */
            [data-testid="stVerticalBlock"] div.stMarkdown h2,
            [data-testid="stVerticalBlock"] div.stMarkdown h3 {
                color: #000000 !important;
                font-weight: 700 !important;
                margin-bottom: 15px !important;
                margin-top: 5px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <h3 style="color: #000000; font-weight: 700; font-size: 22px;">ROI Calculator</h3>
            <p style="color: #000000; font-weight: 500; font-size: 16px; margin-bottom: 20px;">Quantify the financial benefits of operational improvements using our ROI models.</p>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                improvement_type = st.selectbox(
                    "Improvement Category",
                    ["Water Loss Reduction", "Energy Efficiency", "Infrastructure Health", "Compliance"]
                )
                
                if improvement_type == "Water Loss Reduction":
                    current_loss = st.slider("Current Water Loss (%)", 5, 30, 15)
                    target_loss = st.slider("Target Water Loss (%)", 5, current_loss, max(5, current_loss-5))
                    annual_volume = st.number_input("Annual Water Volume (cubic meters)", value=1000000)
                    unit_cost = st.number_input("Cost per Cubic Meter (£)", value=0.5, step=0.1)
                    implementation_cost = st.number_input("Implementation Cost (£)", value=500000)
                elif improvement_type == "Energy Efficiency":
                    current_energy = st.slider("Current Energy Usage (MWh/year)", 100000, 2000000, 1000000, step=50000)
                    target_energy = st.slider("Target Energy Usage (MWh/year)", 100000, current_energy, max(100000, current_energy-100000), step=50000)
                    energy_cost = st.number_input("Energy Cost (£/kWh)", value=0.15, step=0.01)
                    carbon_factor = st.number_input("Carbon Factor (kg CO2e/kWh)", value=0.233, step=0.01)
                    implementation_cost = st.number_input("Implementation Cost (£)", value=300000)
                elif improvement_type == "Infrastructure Health":
                    current_failure = st.slider("Current Failure Rate (%/year)", 1, 20, 5)
                    target_failure = st.slider("Target Failure Rate (%/year)", 1, current_failure, max(1, current_failure-2))
                    failure_cost = st.number_input("Average Cost per Failure (£)", value=5000)
                    asset_count = st.number_input("Number of Assets", value=100, step=10)
                    implementation_cost = st.number_input("Implementation Cost (£)", value=250000)
                else:  # Compliance
                    current_compliance = st.slider("Current Compliance Rate (%)", 70, 99, 90)
                    target_compliance = st.slider("Target Compliance Rate (%)", current_compliance, 100, min(100, current_compliance+5))
                    violation_cost = st.number_input("Annual Violation Costs (£)", value=100000)
                    reputation_value = st.number_input("Reputation Value (£)", value=50000)
                    implementation_cost = st.number_input("Implementation Cost (£)", value=150000)
                    
                timeframe = st.slider("Timeframe (years)", 1, 10, 5)
                
                # Style the button with custom CSS
                st.markdown("""
                <style>
                div.stButton > button {
                    background-color: #FF6900; 
                    color: white; 
                    font-weight: bold; 
                    padding: 0.5rem 1rem; 
                    font-size: 16px; 
                    border-radius: 5px; 
                    border: none;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: all 0.2s ease;
                    width: 100%;
                }
                div.stButton > button:hover {
                    background-color: #E05E00;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                    transform: translateY(-1px);
                }
                div.stButton > button:active {
                    transform: translateY(1px);
                    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Add spacing before button
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                calculate_btn = st.button("Calculate ROI")
            
            with col2:
                if calculate_btn:
                    st.markdown("### ROI Analysis Results")
                    # Call our financial module with the appropriate parameters
                    if improvement_type == "Water Loss Reduction":
                        # Prepare parameters for financial calculation
                        params = {
                            "current_loss": current_loss,
                            "target_loss": target_loss,
                            "annual_volume": annual_volume,
                            "unit_cost": unit_cost,
                            "implementation_cost": implementation_cost,
                            "timeframe": timeframe
                        }
                        
                        # Calculate ROI using our financial component
                        result = calculate_financial_impact("water_loss", params, tenant_id="default")
                        
                        # Extract results
                        savings = result.get("annual_savings", 0)
                        total_savings = result.get("total_savings", 0)
                        net_benefit = result.get("net_benefit", 0)
                        roi_percent = result.get("roi_percentage", 0)
                        payback_years = result.get("payback_period", 0)
                    
                        # Display metrics with improved styling
                        st.markdown("""
                        <style>
                        .metric-card {
                            background-color: white;
                            border-radius: 5px;
                            padding: 15px 20px;
                            margin-bottom: 12px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                            border-left: 4px solid #0366d6;
                        }
                        .metric-name {
                            color: #6c757d;
                            font-size: 0.9rem;
                            font-weight: 500;
                            margin-bottom: 5px;
                        }
                        .metric-value {
                            color: #24292e;
                            font-size: 1.5rem;
                            font-weight: 600;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-name">Annual Savings</div>
                                <div class="metric-value">£{savings:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-name">Net Benefit</div>
                                <div class="metric-value">£{net_benefit:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-name">Total Savings ({timeframe} years)</div>
                                <div class="metric-value">£{total_savings:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid #28a745;">
                                <div class="metric-name">ROI</div>
                                <div class="metric-value">{roi_percent:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 4px solid #f9826c;">
                            <div class="metric-name">Payback Period</div>
                            <div class="metric-value">{payback_years:.1f} years</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success(f"Reducing water loss from {current_loss}% to {target_loss}% provides a positive ROI with {payback_years:.1f} year payback period.")
                    elif improvement_type == "Energy Efficiency":
                        # Prepare parameters for energy efficiency calculation
                        params = {
                            "current_energy": current_energy,
                            "target_energy": target_energy,
                            "energy_cost": energy_cost,
                            "carbon_factor": carbon_factor,
                            "implementation_cost": implementation_cost,
                            "timeframe": timeframe
                        }
                        
                        # Calculate ROI using our financial component
                        result = calculate_financial_impact("energy_efficiency", params, tenant_id="default")
                        
                        # Extract results
                        cost_savings = result.get("annual_savings", 0)
                        carbon_reduction = result.get("carbon_reduction", 0)
                        total_savings = result.get("total_savings", 0)
                        net_benefit = result.get("net_benefit", 0)
                        roi_percent = result.get("roi_percentage", 0)
                        payback_years = result.get("payback_period", 0)
                        
                        # Display metrics with improved styling
                        st.markdown("""
                        <style>
                        .metric-card {
                            background-color: white;
                            border-radius: 5px;
                            padding: 15px 20px;
                            margin-bottom: 12px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                            border-left: 4px solid #0366d6;
                        }
                        .metric-name {
                            color: #6c757d;
                            font-size: 0.9rem;
                            font-weight: 500;
                            margin-bottom: 5px;
                        }
                        .metric-value {
                            color: #24292e;
                            font-size: 1.5rem;
                            font-weight: 600;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-name">Annual Cost Savings</div>
                                <div class="metric-value">£{cost_savings:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-name">Net Benefit</div>
                                <div class="metric-value">£{net_benefit:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid #28a745;">
                                <div class="metric-name">Total Savings ({timeframe} years)</div>
                                <div class="metric-value">£{total_savings:,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid #28a745;">
                                <div class="metric-name">ROI</div>
                                <div class="metric-value">{roi_percent:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 4px solid #f9826c;">
                            <div class="metric-name">Payback Period</div>
                            <div class="metric-value">{payback_years:.1f} years</div>
                        </div>
                        
                        <div class="metric-card" style="border-left: 4px solid #2188ff;">
                            <div class="metric-name">Carbon Reduction</div>
                            <div class="metric-value">{carbon_reduction:.1f} tons CO2/year</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success(f"Energy efficiency improvements provide a positive ROI with {payback_years:.1f} year payback period.")
                    else:
                        # Placeholder for other financial calculations with improved visibility
                        st.info("Financial impact calculation would be performed based on your inputs.")
                        st.markdown("""
                        <div style="background-color:#ffffff; height:230px; border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 15px; position: relative; border: 1px solid #e0e0e0;">
                            <div style="position: absolute; top: 10px; left: 15px; font-weight: 600; color: #24292e; font-size: 14px;">ROI Analysis Chart</div>
                            <div style="display: flex; align-items:center; justify-content:center; height: 100%; width: 100%;">
                                <div style="text-align: center;">
                                    <div style="margin-bottom: 15px;">
                                        <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                            <rect x="3" y="8" width="4" height="12" rx="1" fill="#FFECE3" stroke="#FF6900" stroke-width="2"/>
                                            <rect x="10" y="5" width="4" height="15" rx="1" fill="#FFECE3" stroke="#FF6900" stroke-width="2"/>
                                            <rect x="17" y="3" width="4" height="17" rx="1" fill="#FFECE3" stroke="#FF6900" stroke-width="2"/>
                                            <path d="M3 4L7 4" stroke="#FF6900" stroke-width="2" stroke-linecap="round"/>
                                            <path d="M17 22L21 22" stroke="#FF6900" stroke-width="2" stroke-linecap="round"/>
                                            <path d="M10 22L14 22" stroke="#FF6900" stroke-width="2" stroke-linecap="round"/>
                                            <path d="M3 22L7 22" stroke="#FF6900" stroke-width="2" stroke-linecap="round"/>
                                        </svg>
                                    </div>
                                    <span style="color: #6c757d; font-size: 0.95rem;">Chart appears when financial data is loaded</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Enter your improvement parameters and click 'Calculate ROI' to see the financial impact analysis.")
        except Exception as e:
            st.error(f"Error in Financial Analysis module: {str(e)}")
    
    # 6. Reports tab
    with tabs[5]:
        st.markdown("## Reports")
        try:
            st.markdown("### Generate Custom Reports")
            st.markdown("Create standardized reports for regulatory compliance, operational performance, and stakeholder communication.")
            
            col1, col2 = st.columns(2)
            with col1:
                report_type = st.selectbox(
                    "Report Type",
                    ["Regulatory Compliance", "Operational Performance", "Infrastructure Status", "Financial Summary", "Executive Dashboard"]
                )
                
                time_period = st.selectbox(
                    "Time Period",
                    ["Last Month", "Last Quarter", "Year to Date", "Last 12 Months", "Custom Range"]
                )
                
                if time_period == "Custom Range":
                    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
                    end_date = st.date_input("End Date", value=datetime.now())
                
                output_format = st.selectbox(
                    "Output Format",
                    ["PDF", "Excel", "CSV", "Interactive Dashboard"]
                )
                
                include_sections = st.multiselect(
                    "Include Sections",
                    ["Executive Summary", "Data Visualizations", "Detailed Metrics", "Anomaly Analysis", "Trend Analysis", "Recommendations"],
                    default=["Executive Summary", "Data Visualizations"]
                )
                
                generate_btn = st.button("Generate Report")
            
            with col2:
                if generate_btn:
                    st.success(f"{report_type} report is being generated...")
                    
                    with st.spinner("Preparing report..."):
                        time.sleep(1)  # Simulate processing
                        
                        st.markdown("### Report Preview")
                        st.markdown("""
                        <div style="background-color:white; padding:25px; border:1px solid #e0e0e0; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                <div style="width: 5px; height: 30px; background-color: #FF6900; margin-right: 15px; border-radius: 2px;"></div>
                                <h3 style="color:#000000; margin: 0; font-weight: 700;">Thames Water - Operational Performance Report</h3>
                            </div>
                            <p style="color:#444; background-color: #f8f8f8; padding: 8px 15px; border-radius: 5px; display: inline-block; margin-bottom: 20px;">Reporting Period: April 1, 2025 - May 1, 2025</p>
                            <hr style="border: none; height: 1px; background-color: #e0e0e0; margin: 20px 0;">
                            
                            <h4 style="color:#000000; font-weight: 700; font-size: 18px;">Executive Summary</h4>
                            <p style="color:#000000; line-height: 1.5;">This report summarizes key operational metrics for the specified time period, highlighting performance trends, anomalies, and recommendations.</p>
                            
                            <div style="background-color:#ffffff; height:230px; border-radius:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 15px; position: relative; border: 1px solid #e0e0e0; margin: 25px 0;">
                                <div style="position: absolute; top: 10px; left: 15px; font-weight: 600; color: #24292e; font-size: 14px;">Executive Dashboard</div>
                                <div style="display: flex; align-items:center; justify-content:center; height: 100%; width: 100%;">
                                    <div style="text-align: center;">
                                        <div style="margin-bottom: 15px;">
                                            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                <rect x="3" y="3" width="7" height="7" rx="1" fill="#FFECE3" stroke="#FF6900" stroke-width="2"/>
                                                <rect x="3" y="14" width="7" height="7" rx="1" fill="#FFECE3" stroke="#FF6900" stroke-width="2"/>
                                                <rect x="14" y="3" width="7" height="18" rx="1" fill="#FFECE3" stroke="#FF6900" stroke-width="2"/>
                                            </svg>
                                        </div>
                                        <span style="color: #6c757d; font-size: 0.95rem;">Interactive charts will appear in the final report</span>
                                    </div>
                                </div>
                            </div>
                            
                            <h4 style="color:#000000; font-weight: 700; font-size: 18px;">Key Findings</h4>
                            <ul style="color:#000000; padding-left: 20px; line-height: 1.6;">
                                <li><span style="font-weight: 600; color: #218838;">Overall operational efficiency improved by 3.2%</span></li>
                                <li><span style="font-weight: 600; color: #218838;">Water loss reduced from 12.5% to 11.7%</span></li>
                                <li><span style="font-weight: 600; color: #0366d6;">3 compliance violations resolved, 1 new violation identified</span></li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.download_button(
                            label="Download Report",
                            data="This would be the actual report file",
                            file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.info("Configure your report parameters and click 'Generate Report' to create a custom report.")
                    
                    st.markdown("### Scheduled Reports")
                    st.markdown("""
                    You can set up automatic report generation on a schedule. Reports will be emailed to specified recipients.
                    
                    **Currently Scheduled:**
                    - Monthly Compliance Report (1st of each month)
                    - Weekly Operations Summary (Every Monday)
                    - Quarterly Executive Dashboard (End of each quarter)
                    """)
        except Exception as e:
            st.error(f"Error in Reports module: {str(e)}")


def render_dashboard():
    st.title(f"{st.session_state.client_name} Analytics Dashboard")
    
    # Date filter
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Key Performance Overview")
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now(),
        )
    
    # Summary KPIs - Demo data since no API is required
    # We'll customize this based on client selection
    if st.session_state.client_name == "Thames Water":
        kpi_data = {
            "water_loss": {"value": "12.5%", "status": "amber", "change": "+0.8%"},
            "compliance": {"value": "98.2%", "status": "green", "change": "+1.2%"},
            "energy_usage": {"value": "450 MWh", "status": "red", "change": "+5.4%"},
            "customer_satisfaction": {"value": "4.3/5", "status": "green", "change": "+0.2"},
            "operational_efficiency": {"value": "87.6%", "status": "amber", "change": "-1.1%"},
            "infrastructure_health": {"value": "76.8%", "status": "amber", "change": "-0.3%"},
            "financial_performance": {"value": "£2.4M", "status": "green", "change": "+3.2%"}
        }
    elif st.session_state.client_name == "Southern Water":
        kpi_data = {
            "water_loss": {"value": "10.2%", "status": "green", "change": "-1.5%"},
            "compliance": {"value": "97.5%", "status": "green", "change": "+0.8%"},
            "energy_usage": {"value": "425 MWh", "status": "amber", "change": "+2.1%"},
            "customer_satisfaction": {"value": "4.1/5", "status": "amber", "change": "+0.1"},
            "operational_efficiency": {"value": "89.3%", "status": "green", "change": "+1.7%"},
            "infrastructure_health": {"value": "82.5%", "status": "green", "change": "+2.3%"},
            "financial_performance": {"value": "£2.7M", "status": "green", "change": "+4.5%"}
        }
    elif st.session_state.client_name == "Anglian Water":
        kpi_data = {
            "water_loss": {"value": "14.7%", "status": "red", "change": "+2.1%"},
            "compliance": {"value": "96.8%", "status": "amber", "change": "-0.4%"},
            "energy_usage": {"value": "410 MWh", "status": "green", "change": "-3.2%"},
            "customer_satisfaction": {"value": "4.5/5", "status": "green", "change": "+0.3"},
            "operational_efficiency": {"value": "85.1%", "status": "amber", "change": "-0.8%"},
            "infrastructure_health": {"value": "71.3%", "status": "red", "change": "-2.5%"},
            "financial_performance": {"value": "£2.1M", "status": "amber", "change": "-1.2%"}
        }
    else:  # Yorkshire Water or any other
        kpi_data = {
            "water_loss": {"value": "11.3%", "status": "amber", "change": "-0.3%"},
            "compliance": {"value": "99.1%", "status": "green", "change": "+0.5%"},
            "energy_usage": {"value": "438 MWh", "status": "amber", "change": "+1.2%"},
            "customer_satisfaction": {"value": "4.4/5", "status": "green", "change": "+0.1"},
            "operational_efficiency": {"value": "88.9%", "status": "green", "change": "+0.7%"},
            "infrastructure_health": {"value": "79.2%", "status": "amber", "change": "+1.1%"},
            "financial_performance": {"value": "£2.5M", "status": "green", "change": "+2.8%"}
        }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Water Loss</h3>
            <div class="value">{kpi_data['water_loss']['value']}</div>
            <div class="status status-{kpi_data['water_loss']['status']}">{kpi_data['water_loss']['change']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Operational Efficiency</h3>
            <div class="value">{kpi_data['operational_efficiency']['value']}</div>
            <div class="status status-{kpi_data['operational_efficiency']['status']}">{kpi_data['operational_efficiency']['change']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Compliance Rate</h3>
            <div class="value">{kpi_data['compliance']['value']}</div>
            <div class="status status-{kpi_data['compliance']['status']}">{kpi_data['compliance']['change']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Infrastructure Health</h3>
            <div class="value">{kpi_data['infrastructure_health']['value']}</div>
            <div class="status status-{kpi_data['infrastructure_health']['status']}">{kpi_data['infrastructure_health']['change']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Energy Usage</h3>
            <div class="value">{kpi_data['energy_usage']['value']}</div>
            <div class="status status-{kpi_data['energy_usage']['status']}">{kpi_data['energy_usage']['change']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Customer Satisfaction</h3>
            <div class="value">{kpi_data['customer_satisfaction']['value']}</div>
            <div class="status status-{kpi_data['customer_satisfaction']['status']}">{kpi_data['customer_satisfaction']['change']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Financial Performance</h3>
            <div class="value">{kpi_data['financial_performance']['value']}</div>
            <div class="status status-{kpi_data['financial_performance']['status']}">{kpi_data['financial_performance']['change']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo high priority alerts based on client
        # Thames Water and Anglian Water have alerts, others are normal
        if st.session_state.client_name in ["Thames Water", "Anglian Water"]:
            alert_count = 3 if st.session_state.client_name == "Thames Water" else 5
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 5px solid #FF4B4B;">
                <h3>High Priority Alerts</h3>
                <div class="value">{alert_count}</div>
                <div class="status status-red">Requires Attention</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 5px solid #28A745;">
                <h3>System Status</h3>
                <div class="value">Normal</div>
                <div class="status status-green">No Critical Alerts</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance trends
    st.markdown("### Performance Trends")
    
    tab1, tab2, tab3 = st.tabs(["Operational Metrics", "Financial Analysis", "Compliance"])
    
    with tab1:
        try:
            # Simulated API call for operational metrics
            trends_data = {
                "dates": [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(30, 0, -1)],
                "water_loss": [11.8, 12.1, 12.0, 12.2, 12.3, 12.5, 12.8, 12.7, 12.5, 12.4, 12.6, 12.5, 12.3, 12.4, 12.5, 12.6, 12.8, 12.7, 12.5, 12.3, 12.1, 12.4, 12.5, 12.6, 12.7, 12.5, 12.3, 12.1, 12.0, 12.5],
                "energy_usage": [430, 428, 432, 435, 440, 442, 445, 450, 455, 452, 448, 445, 442, 444, 446, 448, 450, 452, 448, 446, 444, 442, 445, 448, 450, 452, 448, 445, 440, 450],
                "operational_efficiency": [88.5, 88.3, 88.0, 87.8, 87.5, 87.3, 87.0, 86.8, 86.5, 86.3, 86.0, 86.2, 86.5, 86.8, 87.0, 87.2, 87.5, 87.3, 87.0, 86.8, 87.0, 87.2, 87.5, 87.8, 88.0, 87.8, 87.5, 87.3, 87.0, 87.6]
            }
            
            df = pd.DataFrame(trends_data)
            df["dates"] = pd.to_datetime(df["dates"])
            
            fig = go.Figure()
            
            # Water Loss Trend
            fig.add_trace(go.Scatter(
                x=df["dates"], y=df["water_loss"],
                mode='lines',
                name='Water Loss (%)',
                line=dict(color='#00A1D6', width=2)
            ))
            
            # Create a secondary y-axis for energy usage
            fig.add_trace(go.Scatter(
                x=df["dates"], y=df["energy_usage"],
                mode='lines',
                name='Energy Usage (MWh)',
                line=dict(color='#FFB107', width=2),
                yaxis='y2'
            ))
            
            # Add operational efficiency
            fig.add_trace(go.Scatter(
                x=df["dates"], y=df["operational_efficiency"],
                mode='lines',
                name='Operational Efficiency (%)',
                line=dict(color='#28A745', width=2),
                yaxis='y3'
            ))
            
            # Layout with multiple y-axes
            fig.update_layout(
                title='Operational Metrics Trends',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Water Loss (%)', side='left', showgrid=True),
                yaxis2=dict(title='Energy Usage (MWh)', overlaying='y', side='right', showgrid=False),
                yaxis3=dict(title='Efficiency (%)', overlaying='y', anchor='free', position=1.0, showgrid=False),
                margin=dict(l=40, r=40, t=40, b=40),
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating operational trends chart: {str(e)}")
    
    with tab2:
        try:
            # Financial metrics trend
            financial_data = {
                "dates": [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(30, 0, -1)],
                "revenue": [2.2, 2.25, 2.3, 2.28, 2.32, 2.35, 2.4, 2.38, 2.42, 2.45, 2.5, 2.48, 2.52, 2.55, 2.6, 2.58, 2.62, 2.65, 2.7, 2.68, 2.72, 2.75, 2.8, 2.78, 2.82, 2.85, 2.9, 2.88, 2.92, 2.4],
                "costs": [1.8, 1.82, 1.85, 1.87, 1.9, 1.92, 1.95, 1.97, 2.0, 2.02, 2.05, 2.03, 2.0, 1.98, 1.95, 1.97, 2.0, 2.02, 2.05, 2.03, 2.0, 1.98, 1.95, 1.97, 2.0, 2.02, 2.05, 2.03, 2.0, 1.95],
                "profit": [0.4, 0.43, 0.45, 0.41, 0.42, 0.43, 0.45, 0.41, 0.42, 0.43, 0.45, 0.45, 0.52, 0.57, 0.65, 0.61, 0.62, 0.63, 0.65, 0.65, 0.72, 0.77, 0.85, 0.81, 0.82, 0.83, 0.85, 0.85, 0.92, 0.45]
            }
            
            df = pd.DataFrame(financial_data)
            df["dates"] = pd.to_datetime(df["dates"])
            
            fig = go.Figure()
            
            # Revenue line
            fig.add_trace(go.Scatter(
                x=df["dates"], y=df["revenue"],
                mode='lines',
                name='Revenue (£M)',
                line=dict(color='#28A745', width=2)
            ))
            
            # Cost line
            fig.add_trace(go.Scatter(
                x=df["dates"], y=df["costs"],
                mode='lines',
                name='Costs (£M)',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            # Profit as a bar
            fig.add_trace(go.Bar(
                x=df["dates"], y=df["profit"],
                name='Profit (£M)',
                marker_color='#00A1D6'
            ))
            
            fig.update_layout(
                title='Financial Performance',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Amount (£ Millions)'),
                margin=dict(l=40, r=40, t=40, b=40),
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating financial trends chart: {str(e)}")
    
    with tab3:
        try:
            # Compliance metrics
            compliance_data = {
                "categories": ["Water Quality", "Environmental", "Regulatory Reporting", "Safety Standards", "Infrastructure"],
                "compliance_rate": [98.5, 97.2, 99.1, 96.8, 92.3],
                "target": [99.0, 98.0, 100.0, 98.0, 95.0],
                "previous_period": [97.8, 96.5, 98.8, 96.0, 91.2]
            }
            
            df = pd.DataFrame(compliance_data)
            
            fig = go.Figure()
            
            # Current compliance bar
            fig.add_trace(go.Bar(
                x=df["categories"], y=df["compliance_rate"],
                name='Current Compliance (%)',
                marker_color='#00A1D6'
            ))
            
            # Previous period bar
            fig.add_trace(go.Bar(
                x=df["categories"], y=df["previous_period"],
                name='Previous Period (%)',
                marker_color='#005670'
            ))
            
            # Target line
            fig.add_trace(go.Scatter(
                x=df["categories"], y=df["target"],
                mode='lines+markers',
                name='Target (%)',
                line=dict(color='#28A745', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Compliance by Category',
                xaxis=dict(title='Category'),
                yaxis=dict(title='Compliance Rate (%)', range=[90, 100]),
                margin=dict(l=40, r=40, t=40, b=40),
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating compliance chart: {str(e)}")
    
    # Automated Insights
    st.markdown("### Automated Insights")
    
    try:
        # Call generate_insights with proper parameters for demo data
        insights = generate_insights(None, tenant_id="default")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: white; padding: 15px; border-radius: 5px; border-left: 4px solid #0366d6; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #000000; margin-bottom: 15px;">
                <h4 style="color: #000000; font-weight: 600; margin-top: 0; margin-bottom: 10px;">Key Observations</h4>
                <ul style="color: #000000; padding-left: 20px; margin-bottom: 0;">
                    <li>Water loss is trending upward over the past 7 days, with a 0.8% increase.</li>
                    <li>Energy usage is above target by 5.4%, primarily at treatment plants.</li>
                    <li>Customer satisfaction has improved for the third consecutive month.</li>
                    <li>Infrastructure health index declined in the central region.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: white; padding: 15px; border-radius: 5px; border-left: 4px solid #FF6900; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #000000; margin-bottom: 15px;">
                <h4 style="color: #000000; font-weight: 600; margin-top: 0; margin-bottom: 10px;">Recommended Actions</h4>
                <ul style="color: #000000; padding-left: 20px; margin-bottom: 0;">
                    <li><strong style="color: #dc3545;">High Priority:</strong> Investigate increased water loss in the South Catchment area.</li>
                    <li><strong style="color: #ffc107;">Medium Priority:</strong> Review energy consumption at North Treatment plant.</li>
                    <li><strong style="color: #28a745;">Low Priority:</strong> Schedule infrastructure assessment for central region assets.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error generating automated insights: {str(e)}")
    
    # System Integration Status
    st.markdown("### System Integration Status")
    
    try:
        # This would be fetched from the API in production
        integration_status = {
            "SCADA System": {"status": "Connected", "last_update": "2 minutes ago", "health": "good"},
            "Billing System": {"status": "Connected", "last_update": "15 minutes ago", "health": "good"},
            "GIS Data": {"status": "Connected", "last_update": "1 hour ago", "health": "good"},
            "IoT Sensors": {"status": "Partial", "last_update": "5 minutes ago", "health": "warning"},
            "Customer Portal": {"status": "Connected", "last_update": "10 minutes ago", "health": "good"},
            "Laboratory LIMS": {"status": "Disconnected", "last_update": "3 hours ago", "health": "poor"},
            "Asset Management": {"status": "Connected", "last_update": "30 minutes ago", "health": "good"}
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        for i, (system, details) in enumerate(integration_status.items()):
            col = [col1, col2, col3, col4][i % 4]
            
            with col:
                color = "#28A745" if details["health"] == "good" else "#FFB107" if details["health"] == "warning" else "#FF4B4B"
                icon = "✅" if details["health"] == "good" else "⚠️" if details["health"] == "warning" else "❌"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 0.8rem; border-radius: 5px; margin-bottom: 1rem; border-left: 5px solid {color};">
                    <div style="font-size: 0.85rem; color: #6c757d;">System Integration</div>
                    <div style="font-weight: bold; font-size: 1rem;">{system}</div>
                    <div style="margin-top: 0.5rem; display: flex; justify-content: space-between;">
                        <span style="font-size: 0.9rem;">{icon} {details['status']}</span>
                        <span style="font-size: 0.8rem; color: #6c757d;">Updated {details['last_update']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying system integration status: {str(e)}")

if __name__ == "__main__":
    main()
