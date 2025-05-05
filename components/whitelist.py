# Client Theme and Whitelist Management Component

import json
import logging
from typing import Dict, List, Any, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default Thames Water theme
DEFAULT_THEME = {
    "colors": {
        "primary": "#005670",      # Deep blue
        "secondary": "#00A1D6",    # Lighter blue
        "success": "#28A745",      # Green
        "warning": "#FFB107",      # Amber
        "danger": "#FF4B4B",       # Red
        "neutral": "#6c757d",      # Gray
        "light": "#f8f9fa",        # Light gray
        "dark": "#212529",         # Dark gray/black
        "background": "#F0F2F6",   # Light background
        "text": "#31333F"          # Text color
    },
    "fonts": {
        "primary": "Roboto, sans-serif",
        "secondary": "Arial, sans-serif",
        "size_base": "16px",
        "size_heading": "24px",
        "size_label": "14px"
    },
    "logo": {
        "url": None,  # Default logo URL
        "width": "120px",
        "height": "auto",
        "position": "left"
    },
    "borders": {
        "radius": "8px",
        "width": "1px",
        "style": "solid",
        "color": "#e1e1e1"
    },
    "shadows": {
        "card": "0 4px 10px rgba(0, 86, 112, 0.1)",
        "button": "0 2px 5px rgba(0, 0, 0, 0.15)",
        "header": "0 2px 10px rgba(0, 0, 0, 0.1)"
    }
}

# Client theme registry
CLIENT_THEMES = {
    # Thames Water theme
    "thames_water": {
        "colors": {
            "primary": "#005670",      # Thames Water primary blue
            "secondary": "#00A1D6",    # Thames Water secondary blue
            "success": "#28A745",
            "warning": "#FFB107",
            "danger": "#FF4B4B",
            "background": "#F0F2F6", 
            "text": "#31333F"
        },
        "fonts": {
            "primary": "Roboto, sans-serif"
        },
        "logo": {
            "url": None  # Could be set to Thames Water logo URL
        }
    },
    
    # Southern Water theme
    "southern_water": {
        "colors": {
            "primary": "#003C71",      # Dark blue
            "secondary": "#6CACE4",    # Light blue
            "success": "#5CB85C",
            "warning": "#F0AD4E",
            "danger": "#D9534F"
        },
        "fonts": {
            "primary": "Arial, sans-serif"
        },
        "logo": {
            "url": None
        }
    },
    
    # Anglian Water theme
    "anglian_water": {
        "colors": {
            "primary": "#0033A0",      # Anglian blue
            "secondary": "#41B6E6",    # Light blue
            "success": "#00A651",      # Green
            "warning": "#FFC72C",      # Yellow
            "danger": "#ED1C24"        # Red
        },
        "fonts": {
            "primary": "Montserrat, sans-serif"
        },
        "logo": {
            "url": None
        }
    },
    
    # Yorkshire Water theme
    "yorkshire_water": {
        "colors": {
            "primary": "#0072CE",      # Yorkshire blue
            "secondary": "#00AEEF",    # Light blue
            "success": "#009639",      # Green
            "warning": "#FFD100",      # Yellow
            "danger": "#E31B23"        # Red
        },
        "fonts": {
            "primary": "Lato, sans-serif"
        },
        "logo": {
            "url": None
        }
    }
}

# Function to get a theme by client ID
def get_client_theme(client_id: str) -> Dict[str, Any]:
    """Get the theme configuration for a specific client
    
    Args:
        client_id: Client identifier
        
    Returns:
        Theme configuration dictionary
    """
    # Default to Thames Water theme if client not found
    client_theme = CLIENT_THEMES.get(client_id.lower(), CLIENT_THEMES["thames_water"])
    
    # Deep merge with default theme for any missing properties
    return deep_merge(DEFAULT_THEME.copy(), client_theme)

# Function to apply client theme to Streamlit
def apply_client_theme(client_id: str) -> Dict[str, Any]:
    """Apply a client's theme to the Streamlit application
    
    Args:
        client_id: Client identifier
        
    Returns:
        Applied theme configuration
    """
    # Get the client's theme
    theme = get_client_theme(client_id)
    
    # Generate CSS for the theme
    css = generate_theme_css(theme)
    
    # Apply CSS to Streamlit
    try:
        import streamlit as st
        st.markdown(css, unsafe_allow_html=True)
        
        # If logo URL is provided, display it
        if theme["logo"].get("url"):
            logo_width = theme["logo"].get("width", "120px")
            position = theme["logo"].get("position", "left")
            
            # Wrapper div with style based on position
            align = "left" if position == "left" else "center" if position == "center" else "right"
            logo_html = f"""
            <div style="text-align: {align}; margin-bottom: 20px;">
                <img src="{theme['logo']['url']}" width="{logo_width}" />
            </div>
            """
            st.markdown(logo_html, unsafe_allow_html=True)
    except ImportError:
        logger.warning("Streamlit not available, CSS not applied")
    except Exception as e:
        logger.error(f"Error applying theme CSS: {str(e)}")
    
    return theme

# Helper function to generate CSS from theme
def generate_theme_css(theme: Dict[str, Any]) -> str:
    """Generate CSS for a theme configuration
    
    Args:
        theme: Theme configuration dictionary
        
    Returns:
        CSS string
    """
    colors = theme.get("colors", {})
    fonts = theme.get("fonts", {})
    borders = theme.get("borders", {})
    shadows = theme.get("shadows", {})
    
    css = f"""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        /* General Styles */
        body, .stApp {{
            font-family: {fonts.get('primary', 'Roboto, sans-serif')};
            color: {colors.get('text', '#31333F')};
            background-color: {colors.get('background', '#F0F2F6')};
        }}
        .stApp > header {{
            background-color: {colors.get('primary', '#005670')};
            color: white;
        }}
        /* Metric Card Styles */
        .kpi-card {{
            background-color: white; 
            padding: 1rem 1rem; 
            border-radius: {borders.get('radius', '8px')};
            box-shadow: {shadows.get('card', '0 4px 10px rgba(0, 86, 112, 0.1)')}; 
            text-align: center;
            border-left: 5px solid {colors.get('secondary', '#00A1D6')}; 
            margin-bottom: 1rem;
            min-height: 120px; 
            display: flex; 
            flex-direction: column; 
            justify-content: space-between;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }}
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 86, 112, 0.15);
        }}
        /* Status Indicators */
        .status-good {{
            color: {colors.get('success', '#28A745')};
        }}
        .status-warning {{
            color: {colors.get('warning', '#FFB107')};
        }}
        .status-critical {{
            color: {colors.get('danger', '#FF4B4B')};
        }}
        /* Navigation and Buttons */
        .stButton button {{
            background-color: {colors.get('primary', '#005670')};
            color: white;
            border-radius: {borders.get('radius', '8px')};
            border: none;
            padding: 0.5rem 1rem;
            box-shadow: {shadows.get('button', '0 2px 5px rgba(0, 0, 0, 0.15)')};
        }}
        .stButton button:hover {{
            background-color: {colors.get('secondary', '#00A1D6')};
        }}
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: {borders.get('radius', '8px')} {borders.get('radius', '8px')} 0px 0px;
            padding: 10px 20px;
            background-color: #f0f2f6;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {colors.get('primary', '#005670')};
            color: white;
        }}
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {colors.get('background', '#F0F2F6')};
            border-right: {borders.get('width', '1px')} {borders.get('style', 'solid')} {borders.get('color', '#e1e1e1')};
        }}
        /* Alert Box Styling */
        .alert-box {{
            padding: 1rem;
            border-radius: {borders.get('radius', '8px')};
            margin-bottom: 1rem;
            font-weight: 500;
        }}
        .alert-box.info {{
            background-color: {colors.get('secondary', '#00A1D6')}25;
            border-left: 4px solid {colors.get('secondary', '#00A1D6')};
        }}
        .alert-box.warning {{
            background-color: {colors.get('warning', '#FFB107')}25;
            border-left: 4px solid {colors.get('warning', '#FFB107')};
        }}
        .alert-box.danger {{
            background-color: {colors.get('danger', '#FF4B4B')}25;
            border-left: 4px solid {colors.get('danger', '#FF4B4B')};
        }}
        .alert-box.success {{
            background-color: {colors.get('success', '#28A745')}25;
            border-left: 4px solid {colors.get('success', '#28A745')};
        }}
        /* Custom Headings */
        .section-heading {{
            color: {colors.get('primary', '#005670')};
            font-size: {fonts.get('size_heading', '24px')};
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {colors.get('secondary', '#00A1D6')};
        }}
        /* Custom table styling */
        .stDataFrame table {{
            border-collapse: separate;
            border-spacing: 0;
            border-radius: {borders.get('radius', '8px')};
            overflow: hidden;
            box-shadow: {shadows.get('card', '0 4px 10px rgba(0, 86, 112, 0.1)')};
        }}
        .stDataFrame th {{
            background-color: {colors.get('primary', '#005670')};
            color: white;
            font-weight: 500;
            padding: 12px 15px;
            text-align: left;
        }}
        .stDataFrame td {{
            padding: 10px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .stDataFrame tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .stDataFrame tr:hover {{
            background-color: #f0f0f0;
        }}
        /* Loading Animation */
        .loading-spinner {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }}
        .loading-spinner:after {{
            content: " ";
            display: block;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 6px solid {colors.get('primary', '#005670')};
            border-color: {colors.get('primary', '#005670')} transparent;
            animation: spinner 1.2s linear infinite;
        }}
        @keyframes spinner {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    
    return css

# Helper function to deep merge dictionaries
def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge into base (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Replace or add value
            result[key] = value
            
    return result

# Function to register a new client theme
def register_client_theme(client_id: str, theme: Dict[str, Any]) -> bool:
    """Register a new client theme or update an existing one
    
    Args:
        client_id: Client identifier
        theme: Theme configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate theme structure
        if not isinstance(theme, dict):
            logger.error(f"Invalid theme format for client {client_id}")
            return False
            
        # Normalize client ID
        client_id = client_id.lower().strip()
        
        # Register the theme
        CLIENT_THEMES[client_id] = theme
        logger.info(f"Registered theme for client {client_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error registering theme for client {client_id}: {str(e)}")
        return False

# Function to generate markdown with theme-specific styling
def styled_markdown(text: str, style: str = "info", theme: Dict[str, Any] = None) -> str:
    """Generate styled markdown text based on client theme
    
    Args:
        text: Markdown text to style
        style: Style to apply (info, success, warning, danger)
        theme: Theme configuration (uses default if None)
        
    Returns:
        HTML-formatted text with styles applied
    """
    if theme is None:
        theme = DEFAULT_THEME
        
    colors = theme.get("colors", {})
    
    # Select color based on style
    if style == "success":
        color = colors.get("success", "#28A745")
    elif style == "warning":
        color = colors.get("warning", "#FFB107")
    elif style == "danger":
        color = colors.get("danger", "#FF4B4B")
    else:  # default to info
        color = colors.get("secondary", "#00A1D6")
    
    # Create styled HTML
    html = f"""
    <div class="alert-box {style}">
        {text}
    </div>
    """
    
    return html

# For testing
if __name__ == "__main__":
    # Test theme generation
    thames_theme = get_client_theme("thames_water")
    theme_css = generate_theme_css(thames_theme)
    
    print("Theme CSS generated successfully")
    
    # Test different client themes
    for client in CLIENT_THEMES.keys():
        theme = get_client_theme(client)
        print(f"Retrieved theme for {client} with primary color: {theme['colors']['primary']}")
