import streamlit as st
import json

def fix_all_styling():
    # Apply direct CSS for styling tabs (most aggressive approach)
    st.markdown("""
    <style>
    /* Force tab styling */
    div[data-testid="stTabs"] > div > div > div[role="tablist"] button[aria-selected="true"] {
        background-color: #FF6900 !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    div[data-testid="stTabs"] > div > div > div[role="tablist"] button {
        background-color: #f0f0f0 !important;
        color: #333333 !important;
        border-radius: 8px 8px 0 0 !important;
        margin-right: 4px !important;
    }
    
    /* Make text in tabs visible */
    div[data-testid="stTabs"] button p,
    div[data-testid="stTabs"] button div,
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #333333 !important;
    }
    
    /* Make text in active tab white */
    div[data-testid="stTabs"] button[aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] div,
    div[data-testid="stTabs"] button[aria-selected="true"][data-baseweb="tab"] {
        color: white !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #2C3E50 !important;
        color: white !important;
    }
    
    section[data-testid="stSidebar"] div {  
        color: white !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* Button styling */
    button, div.stButton > button, div.stDownloadButton > button {
        background-color: #FF6900 !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Override Streamlit's theme by directly modifying the config.toml file's contents
    # (won't be saved to disk but applies to current session)
    theme_config = {
        "primaryColor": "#FF6900",
        "backgroundColor": "#F8F9FA",
        "secondaryBackgroundColor": "#2C3E50",
        "textColor": "#333333",
        "font": "sans serif"
    }
    
    # Apply by creating a custom theme object and injecting it into Streamlit
    try:
        # This relies on internals and may not work in all Streamlit versions
        if hasattr(st, '_config'):
            if hasattr(st._config, '_config_options'):
                # Try to access the config options
                config_options = getattr(st._config, '_config_options')
                if 'theme' in config_options:
                    # Update theme settings
                    for key, value in theme_config.items():
                        config_options['theme'][key] = value
    except Exception as e:
        # Silently continue if this approach doesn't work
        pass
