# FastAPI Backend for Thames Water Enterprise Analytics Platform
# Main API application entry point

from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import uuid
import logging

# Local imports
from . import models, database, auth
from .routers import client, ingest, analytics
from .connectors import base, scada, gis, billing
from .validators import validate_data
from .transformers import transform_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Database setup
engine = database.get_engine()
models.Base.metadata.create_all(bind=engine)

# Create the FastAPI app
app = FastAPI(
    title="Water Utility Analytics API",
    description="Enterprise API for water utility data integration and analytics",
    version="1.0.0"
)

# Configure CORS for communication with the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from different modules
app.include_router(client.router, prefix="/client", tags=["Client Management"])
app.include_router(ingest.router, prefix="/ingest", tags=["Data Ingestion"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])

# Authentication dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Database dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication routes
@app.post("/auth/token", tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate user and generate access token"""
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get or create tenant ID based on the provided tenant name (form_data.client_id)
    tenant = db.query(models.Tenant).filter(models.Tenant.name == form_data.client_id).first()
    
    if not tenant:
        # For demo purposes only - in production, tenant creation would be a separate admin process
        tenant = models.Tenant(
            id=str(uuid.uuid4()),
            name=form_data.client_id,
            settings=json.dumps({
                "primary_color": "#005670",
                "secondary_color": "#00A1D6",
                "logo_url": None
            })
        )
        db.add(tenant)
        db.commit()
    
    # Create access token
    access_token = auth.create_access_token(
        data={"sub": user.username, "tenant_id": tenant.id}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "tenant_id": tenant.id
    }

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Check system health and component statuses"""
    try:
        # Check database connectivity
        db = database.SessionLocal()
        db_status = "Connected"
        db.close()
    except Exception as e:
        db_status = f"Error: {str(e)}"
    
    # Check other system components
    component_statuses = {
        "database": db_status,
        "api": "Running",
        "version": "1.0.0"
    }
    
    return {
        "status": "healthy" if db_status == "Connected" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": component_statuses
    }

# Alert endpoints
@app.get("/alerts/high-priority", tags=["Alerts"])
async def get_high_priority_alerts(
    tenant_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """Get high priority alerts for a specific tenant"""
    # Verify user has access to this tenant
    auth.verify_token_tenant(token, tenant_id)
    
    try:
        # Query for high priority alerts
        alerts = db.query(models.Alert).filter(
            models.Alert.tenant_id == tenant_id,
            models.Alert.priority == "high",
            models.Alert.resolved == False
        ).all()
        
        return [
            {
                "id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "priority": alert.priority,
                "category": alert.category,
                "created_at": alert.created_at.isoformat()
            }
            for alert in alerts
        ]
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving alerts: {str(e)}"
        )

# Theme endpoint
@app.get("/client/{tenant_id}/theme", tags=["Client Management"])
async def get_client_theme(
    tenant_id: str,
    db: Session = Depends(get_db)
):
    """Get client theme settings based on tenant ID"""
    try:
        tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
        
        if not tenant:
            # Return default theme if tenant not found
            return {
                "primary_color": "#005670",
                "secondary_color": "#00A1D6",
                "success_color": "#28A745",
                "warning_color": "#FFB107",
                "danger_color": "#FF4B4B",
                "logo_url": None
            }
        
        # Parse settings from JSON
        settings = json.loads(tenant.settings)
        
        # Ensure all required theme properties exist
        theme = {
            "primary_color": settings.get("primary_color", "#005670"),
            "secondary_color": settings.get("secondary_color", "#00A1D6"),
            "success_color": settings.get("success_color", "#28A745"),
            "warning_color": settings.get("warning_color", "#FFB107"),
            "danger_color": settings.get("danger_color", "#FF4B4B"),
            "logo_url": settings.get("logo_url")
        }
        
        return theme
    except Exception as e:
        logger.error(f"Error retrieving client theme: {str(e)}")
        # Return default theme on error
        return {
            "primary_color": "#005670",
            "secondary_color": "#00A1D6",
            "success_color": "#28A745",
            "warning_color": "#FFB107",
            "danger_color": "#FF4B4B",
            "logo_url": None
        }

# KPI Summary endpoint
@app.get("/analytics/kpi-summary", tags=["Analytics"])
async def get_kpi_summary(
    tenant_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get KPI summary data for dashboard"""
    try:
        # In a real implementation, this would query the database based on tenant_id and date range
        # For this example, we'll return mock data
        
        # Check if we have any KPI data stored for this tenant
        kpi_records = db.query(models.KpiValue).filter(
            models.KpiValue.tenant_id == tenant_id
        ).order_by(models.KpiValue.date.desc()).limit(30).all()
        
        if kpi_records and len(kpi_records) > 0:
            # If we have real data, use the most recent values
            latest_record = kpi_records[0]
            values = json.loads(latest_record.values)
            
            # Calculate changes from previous period if available
            if len(kpi_records) > 1:
                previous_record = kpi_records[1]
                previous_values = json.loads(previous_record.values)
                
                # Calculate changes for each KPI
                for kpi in values.keys():
                    if kpi in previous_values:
                        # Extract numeric part of value for comparison
                        current = float(values[kpi]["value"].split("%")[0]) if "%" in values[kpi]["value"] else float(values[kpi]["value"].split(" ")[0])
                        prev = float(previous_values[kpi]["value"].split("%")[0]) if "%" in previous_values[kpi]["value"] else float(previous_values[kpi]["value"].split(" ")[0])
                        
                        change = current - prev
                        values[kpi]["change"] = f"{'+' if change >= 0 else ''}{change:.1f}%"
            
            return values
        else:
            # Return mock data if no records exist
            return {
                "water_loss": {"value": "12.5%", "status": "amber", "change": "+0.8%"},
                "compliance": {"value": "98.2%", "status": "green", "change": "+1.2%"},
                "energy_usage": {"value": "450 MWh", "status": "red", "change": "+5.4%"},
                "customer_satisfaction": {"value": "4.3/5", "status": "green", "change": "+0.2"},
                "operational_efficiency": {"value": "87.6%", "status": "amber", "change": "-1.1%"},
                "infrastructure_health": {"value": "76.8%", "status": "amber", "change": "-0.3%"},
                "financial_performance": {"value": "£2.4M", "status": "green", "change": "+3.2%"}
            }
        
    except Exception as e:
        logger.error(f"Error retrieving KPI summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving KPI summary: {str(e)}"
        )

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
