# Arcadis Analytics Dashboard

An enterprise-grade water utility analytics platform designed for comprehensive infrastructure optimization and regulatory reporting, with advanced simulation and compliance capabilities.

## Deployment Instructions

### Prerequisites
- Python 3.11 or higher
- Streamlit

### Local Deployment
1. Clone or download this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

### Streamlit Cloud Deployment
1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy by selecting this repository and specifying `app.py` as the main file

## Features
- Advanced geospatial analytics
- Interactive data visualization
- Financial impact modeling
- Regulatory compliance reporting
- Infrastructure health monitoring
- Data quality assessment

## Configuration
The dashboard is pre-configured to run with the following settings in `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

For database connections, you'll need to set up your own environment variables or update the connection settings in `api/database.py`.
