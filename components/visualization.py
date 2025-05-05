# Data Visualization Components

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thames Water Colors (can be customized for different clients)
DEFAULT_COLORS = {
    "primary": "#005670",      # Deep blue
    "secondary": "#00A1D6",    # Lighter blue
    "success": "#28A745",      # Green
    "warning": "#FFB107",      # Amber
    "danger": "#FF4B4B",       # Red
    "neutral": "#6c757d",      # Gray
    "light": "#f8f9fa",        # Light gray
    "dark": "#212529"          # Dark gray/black
}

# Create custom Plotly template
def create_client_template(colors=None):
    """Create a custom Plotly template with client branding"""
    if colors is None:
        colors = DEFAULT_COLORS
        
    template = go.layout.Template()
    template.layout.font = dict(family="Roboto, sans-serif", size=12, color=colors["dark"])
    template.layout.title.font = dict(family="Roboto, sans-serif", size=16, color=colors["primary"])
    template.layout.paper_bgcolor = 'rgba(0,0,0,0)'  # Transparent background
    template.layout.plot_bgcolor = 'rgba(0,0,0,0)'
    template.layout.margin = dict(l=40, r=40, t=60, b=40)
    template.layout.colorway = [
        colors["secondary"], colors["primary"], colors["success"],
        colors["warning"], colors["danger"], colors["neutral"]
    ]
    template.layout.hoverlabel = dict(bgcolor="white", font_size=12, font_family="Roboto, sans-serif")
    template.layout.xaxis = dict(
        showgrid=False, linecolor=colors["neutral"], tickcolor=colors["neutral"],
        title_font_size=13, tickfont_size=11
    )
    template.layout.yaxis = dict(
        showgrid=True, gridcolor='#e1e1e1', linecolor=colors["neutral"], tickcolor=colors["neutral"],
        title_font_size=13, tickfont_size=11
    )
    template.layout.legend = dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor='rgba(255,255,255,0.6)', font_size=11
    )
    template.layout.title.x = 0.5
    template.layout.title.xanchor = 'center'
    
    return template

# Set default template
client_template = create_client_template()

# Function to create a KPI indicator
def create_kpi_indicator(value, title, target=None, previous=None, format_str="{:,.2f}", increasing_is_good=True, custom_colors=None):
    """Create a KPI gauge indicator with comparison to target and previous value
    
    Args:
        value: Current value of the KPI
        title: Title/name of the KPI
        target: Target value (optional)
        previous: Previous value for comparison (optional)
        format_str: String format for the value display
        increasing_is_good: Whether an increasing value is good (affects color coding)
        custom_colors: Custom color dict (optional)
        
    Returns:
        Plotly figure object
    """
    colors = custom_colors if custom_colors else DEFAULT_COLORS
    
    # Determine color based on target comparison
    if target is not None:
        if increasing_is_good:
            color = colors["success"] if value >= target else colors["danger"]
        else:
            color = colors["success"] if value <= target else colors["danger"]
    else:
        color = colors["secondary"]
    
    # Calculate percentage of target achieved
    if target is not None:
        if target != 0:  # Avoid division by zero
            if increasing_is_good:
                percentage = min(100, (value / target) * 100)
            else:
                # For metrics where lower is better (e.g., costs, losses)
                percentage = min(100, (target / value) * 100) if value != 0 else 100
        else:
            percentage = 100 if (increasing_is_good and value > 0) or (not increasing_is_good and value == 0) else 0
    else:
        percentage = 50  # Default when no target is provided
        
    # Create gauge chart
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"valueformat": format_str, "font": {"size": 20}},
        delta={"reference": previous, "valueformat": format_str, "increasing": {"color": colors["success"] if increasing_is_good else colors["danger"]}, "decreasing": {"color": colors["danger"] if increasing_is_good else colors["success"]}},
        gauge={
            "axis": {"range": [0, target * 1.5] if target is not None else [0, value * 2], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "lightgray",
            "steps": [
                {"range": [0, target], "color": "#e6f2ff"} if target is not None else {"range": [0, value], "color": "#e6f2ff"}
            ],
            "threshold": {
                "line": {"color": colors["primary"], "width": 4},
                "thickness": 0.75,
                "value": target if target is not None else value
            }
        },
        title={"text": title, "font": {"size": 16}}
    ))
    
    # Update layout
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto, sans-serif")
    )
    
    return fig

# Function to create a trend chart
def create_trend_chart(df, x_col, y_cols, title="Trend Analysis", chart_type="line",
                      color_discrete_map=None, template=None, custom_colors=None,
                      filters=None, aggregation=None, resample=None):
    """Create a trend chart for one or more metrics over time
    
    Args:
        df: Pandas DataFrame containing the data
        x_col: Column name for the x-axis (typically time/date)
        y_cols: List of column names for y-axis values
        title: Chart title
        chart_type: Type of chart (line, bar, area)
        color_discrete_map: Manual color mapping for series
        template: Plotly template to use
        custom_colors: Custom colors dictionary
        filters: Dictionary of filters to apply to the data {column: value}
        aggregation: Aggregation method for y values ('sum', 'mean', etc.)
        resample: Time period for resampling time series data ('D', 'W', 'M', etc.)
        
    Returns:
        Plotly figure object
    """
    if template is None:
        template = client_template
    
    colors = custom_colors if custom_colors else DEFAULT_COLORS
    
    # Apply filters if provided
    if filters:
        filtered_df = df.copy()
        for col, val in filters.items():
            if col in filtered_df.columns:
                if isinstance(val, list):
                    filtered_df = filtered_df[filtered_df[col].isin(val)]
                else:
                    filtered_df = filtered_df[filtered_df[col] == val]
    else:
        filtered_df = df.copy()
    
    # Check if x_col is datetime and convert if needed
    if x_col in filtered_df.columns and not pd.api.types.is_datetime64_dtype(filtered_df[x_col]):
        try:
            filtered_df[x_col] = pd.to_datetime(filtered_df[x_col])
        except Exception as e:
            logger.warning(f"Could not convert {x_col} to datetime: {str(e)}")
    
    # Resample time series if needed
    if resample and x_col in filtered_df.columns:
        # Set the datetime column as index
        filtered_df = filtered_df.set_index(x_col)
        
        # Determine aggregation method
        agg_method = aggregation if aggregation else 'mean'
        agg_dict = {col: agg_method for col in y_cols if col in filtered_df.columns}
        
        # Resample and aggregate
        if agg_dict:
            try:
                filtered_df = filtered_df.resample(resample).agg(agg_dict).reset_index()
            except Exception as e:
                logger.error(f"Error resampling data: {str(e)}")
                # Reset index if resampling failed
                if x_col not in filtered_df.columns:
                    filtered_df = filtered_df.reset_index()
        else:
            # Reset index if no columns to aggregate
            filtered_df = filtered_df.reset_index()
    
    # Create appropriate chart type
    if chart_type == "line":
        fig = px.line(
            filtered_df, x=x_col, y=y_cols, title=title,
            color_discrete_map=color_discrete_map,
            template=template
        )
        
        # Add markers to the lines
        for trace in fig.data:
            trace.update(mode='lines+markers', marker=dict(size=6))
            
    elif chart_type == "bar":
        fig = px.bar(
            filtered_df, x=x_col, y=y_cols, title=title,
            color_discrete_map=color_discrete_map,
            template=template,
            barmode="group"
        )
        
    elif chart_type == "area":
        fig = px.area(
            filtered_df, x=x_col, y=y_cols, title=title,
            color_discrete_map=color_discrete_map,
            template=template
        )
        
    else:  # Default to line
        fig = px.line(
            filtered_df, x=x_col, y=y_cols, title=title,
            color_discrete_map=color_discrete_map,
            template=template
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title="Value",
        legend_title="Metrics",
        height=400,
        hovermode="x unified"
    )
    
    # Add range slider for time series
    if x_col in filtered_df.columns and pd.api.types.is_datetime64_dtype(filtered_df[x_col]):
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
    
    return fig

# Function to create a map visualization
def create_geo_map(df, lat_col, lon_col, color_col=None, size_col=None, hover_data=None, zoom=9,
                  mapbox_style="carto-positron", title="Geographic Distribution", custom_colors=None):
    """Create a geographic map visualization of point data
    
    Args:
        df: Pandas DataFrame containing the data
        lat_col: Column name containing latitude values
        lon_col: Column name containing longitude values
        color_col: Column name for color encoding (optional)
        size_col: Column name for marker size encoding (optional)
        hover_data: List of columns to include in hover tooltip
        zoom: Initial zoom level
        mapbox_style: Mapbox style to use
        title: Chart title
        custom_colors: Custom colors dictionary
        
    Returns:
        Plotly figure object
    """
    colors = custom_colors if custom_colors else DEFAULT_COLORS
    
    # Check for required columns
    if lat_col not in df.columns or lon_col not in df.columns:
        logger.error(f"Required geographic columns missing: {lat_col}, {lon_col}")
        # Return an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text="Geographic data columns not found", showarrow=False, font=dict(size=14, color=colors["danger"]))
        fig.update_layout(title=title, height=500)
        return fig
    
    # Create color sequence based on client colors
    color_sequence = [colors["secondary"], colors["primary"], colors["success"], colors["warning"], colors["danger"]]
    
    # Prepare hover data
    if hover_data is None:
        # Default hover data includes coordinates and color/size columns if provided
        hover_data = [lat_col, lon_col]
        if color_col and color_col not in hover_data:
            hover_data.append(color_col)
        if size_col and size_col not in hover_data:
            hover_data.append(size_col)
    
    # Create map
    try:
        fig = px.scatter_mapbox(
            df, 
            lat=lat_col, 
            lon=lon_col,
            color=color_col,
            size=size_col,
            hover_name=color_col if color_col else None,
            hover_data=hover_data,
            zoom=zoom,
            height=600,
            title=title,
            color_discrete_sequence=color_sequence,
            opacity=0.7
        )
        
        # Update layout
        fig.update_layout(
            mapbox_style=mapbox_style,
            margin={"r":0,"t":50,"l":0,"b":0},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating geo map: {str(e)}")
        # Return an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating map: {str(e)}", showarrow=False, font=dict(size=14, color=colors["danger"]))
        fig.update_layout(title=title, height=500)
        return fig

# Function to create a heatmap for correlation analysis
def create_correlation_heatmap(df, columns=None, title="Correlation Analysis", custom_colors=None):
    """Create a heatmap of correlations between numeric variables
    
    Args:
        df: Pandas DataFrame containing the data
        columns: List of columns to include in correlation analysis (defaults to all numeric)
        title: Chart title
        custom_colors: Custom colors dictionary
        
    Returns:
        Plotly figure object
    """
    colors = custom_colors if custom_colors else DEFAULT_COLORS
    
    # Use only numeric columns if none specified
    if columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_columns
    else:
        # Filter to numeric columns that exist in the dataframe
        columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(columns) < 2:
        logger.warning("Not enough numeric columns for correlation analysis")
        # Return an empty figure with warning message
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric variables for correlation analysis", showarrow=False, font=dict(size=14, color=colors["warning"]))
        fig.update_layout(title=title, height=500)
        return fig
    
    try:
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[
                colors["danger"],     # Strong negative correlation
                colors["light"],     # Neutral
                colors["success"]    # Strong positive correlation
            ],
            zmin=-1, zmax=1,
            title=title
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title="",
            yaxis_title="",
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0", "0.5", "1.0"]
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        # Return an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating correlation heatmap: {str(e)}", showarrow=False, font=dict(size=14, color=colors["danger"]))
        fig.update_layout(title=title, height=500)
        return fig

# Function to create a combination chart (e.g., line + bar)
def create_combo_chart(df, x_col, y_cols_left, y_cols_right=None, chart_types=None,
                      title="Combined Metrics", custom_colors=None):
    """Create a combination chart with multiple chart types and y-axes
    
    Args:
        df: Pandas DataFrame containing the data
        x_col: Column name for the x-axis (typically time/date)
        y_cols_left: List of column names for left y-axis
        y_cols_right: List of column names for right y-axis (optional)
        chart_types: Dictionary mapping columns to chart types {'col_name': 'line'/'bar'}
        title: Chart title
        custom_colors: Custom colors dictionary
        
    Returns:
        Plotly figure object
    """
    colors = custom_colors if custom_colors else DEFAULT_COLORS
    
    # Default chart types
    if chart_types is None:
        chart_types = {}
        # Default: columns on left axis are lines, columns on right axis are bars
        for col in y_cols_left:
            chart_types[col] = "line"
        if y_cols_right:
            for col in y_cols_right:
                chart_types[col] = "bar"
    
    # Create subplot with shared x-axis and two y-axes if needed
    fig = make_subplots(specs=[[{"secondary_y": True if y_cols_right else False}]])
    
    # Add traces for left y-axis
    color_index = 0
    color_list = [colors["secondary"], colors["primary"], colors["success"], colors["warning"], colors["danger"], colors["neutral"]]
    
    for i, col in enumerate(y_cols_left):
        col_color = color_list[color_index % len(color_list)]
        color_index += 1
        
        if chart_types.get(col, "line") == "line":
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    mode="lines+markers",
                    line=dict(color=col_color, width=2),
                    marker=dict(size=6)
                ),
                secondary_y=False
            )
        elif chart_types.get(col, "line") == "bar":
            fig.add_trace(
                go.Bar(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    marker_color=col_color,
                    opacity=0.7
                ),
                secondary_y=False
            )
        elif chart_types.get(col, "line") == "area":
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    mode="lines",
                    line=dict(color=col_color, width=2),
                    fill="tozeroy",
                    fillcolor=col_color + "20"  # Add transparency
                ),
                secondary_y=False
            )
    
    # Add traces for right y-axis if provided
    if y_cols_right:
        for i, col in enumerate(y_cols_right):
            col_color = color_list[color_index % len(color_list)]
            color_index += 1
            
            if chart_types.get(col, "bar") == "line":
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col],
                        y=df[col],
                        name=col,
                        mode="lines+markers",
                        line=dict(color=col_color, width=2, dash="dash"),
                        marker=dict(size=6)
                    ),
                    secondary_y=True
                )
            elif chart_types.get(col, "bar") == "bar":
                fig.add_trace(
                    go.Bar(
                        x=df[x_col],
                        y=df[col],
                        name=col,
                        marker_color=col_color,
                        opacity=0.7
                    ),
                    secondary_y=True
                )
            elif chart_types.get(col, "bar") == "area":
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col],
                        y=df[col],
                        name=col,
                        mode="lines",
                        line=dict(color=col_color, width=2, dash="dash"),
                        fill="tozeroy",
                        fillcolor=col_color + "20"  # Add transparency
                    ),
                    secondary_y=True
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        hovermode="x unified",
        template=client_template
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Value", secondary_y=False)
    if y_cols_right:
        fig.update_yaxes(title_text="Value (Secondary)", secondary_y=True)
    
    # Add range slider for time series
    if x_col in df.columns and pd.api.types.is_datetime64_dtype(df[x_col]):
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            )
        )
    
    return fig

# For testing
if __name__ == "__main__":
    # Generate sample data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample time series data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100)
    
    # Trend component
    trend = np.linspace(0, 10, 100)
    
    # Seasonal component
    seasonal = 5 * np.sin(np.linspace(0, 2*np.pi, 100))
    
    # Create base signal
    signal = trend + seasonal
    
    # Add noise
    noise = np.random.normal(0, 1, 100)
    
    # Final time series
    values = signal + noise
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'cost': values * 2.5 + np.random.normal(0, 2, 100),  # Correlated with value
        'efficiency': 100 - (values * 3) + np.random.normal(0, 3, 100),  # Negatively correlated
        'usage': np.random.normal(50, 10, 100),  # Independent variable
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Test KPI indicator
    kpi_fig = create_kpi_indicator(
        value=85.2,
        title="Operational Efficiency",
        target=90,
        previous=82.7,
        format_str="{:.1f}%",
        increasing_is_good=True
    )
    
    # Test trend chart
    trend_fig = create_trend_chart(
        df=df,
        x_col='date',
        y_cols=['value', 'efficiency'],
        title="Key Metrics Trend",
        chart_type="line"
    )
    
    # Test combo chart
    combo_fig = create_combo_chart(
        df=df,
        x_col='date',
        y_cols_left=['value', 'efficiency'],
        y_cols_right=['usage'],
        chart_types={'value': 'line', 'efficiency': 'line', 'usage': 'bar'},
        title="Combined Metrics Analysis"
    )
    
    # Test correlation heatmap
    corr_fig = create_correlation_heatmap(
        df=df,
        columns=['value', 'cost', 'efficiency', 'usage'],
        title="Correlation Analysis"
    )
    
    # Print that tests have been run
    print("Visualization components are ready")
