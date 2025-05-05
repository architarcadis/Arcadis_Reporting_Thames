# SQLAlchemy models for the water analytics application

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .database import Base

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    tenant_id = Column(String, ForeignKey("tenants.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="users")

class Tenant(Base):
    """Tenant model for multi-tenancy support"""
    __tablename__ = "tenants"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, index=True)
    subscription_tier = Column(String, default="standard")
    settings = Column(Text)  # JSON string for tenant settings
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="tenant")
    datasources = relationship("DataSource", back_populates="tenant")
    kpi_values = relationship("KpiValue", back_populates="tenant")
    alerts = relationship("Alert", back_populates="tenant")
    reports = relationship("Report", back_populates="tenant")

class DataSource(Base):
    """Data source configuration for external systems"""
    __tablename__ = "datasources"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"))
    name = Column(String)
    type = Column(String)  # e.g., "scada", "billing", "gis"
    connection_string = Column(String, nullable=True)
    api_key = Column(String, nullable=True)
    settings = Column(Text)  # JSON string for source-specific settings
    is_active = Column(Boolean, default=True)
    last_sync = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="datasources")
    datasets = relationship("Dataset", back_populates="datasource")

class Dataset(Base):
    """Dataset stored from external sources"""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    datasource_id = Column(String, ForeignKey("datasources.id"))
    name = Column(String)
    description = Column(String, nullable=True)
    schema = Column(Text)  # JSON schema of the dataset
    last_updated = Column(DateTime, default=datetime.utcnow)
    record_count = Column(Integer, default=0)
    
    # Relationships
    datasource = relationship("DataSource", back_populates="datasets")
    
class KpiDefinition(Base):
    """KPI definitions"""
    __tablename__ = "kpi_definitions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    description = Column(String, nullable=True)
    calculation_method = Column(String)  # SQL or formula to calculate KPI
    unit = Column(String, nullable=True)  # e.g., "%", "MWh", "£"
    target_value = Column(Float, nullable=True)
    warning_threshold = Column(Float, nullable=True)
    critical_threshold = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class KpiValue(Base):
    """Stored KPI values for tenants"""
    __tablename__ = "kpi_values"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"))
    date = Column(DateTime, default=datetime.utcnow)
    values = Column(Text)  # JSON string of KPI values
    
    # Relationships
    tenant = relationship("Tenant", back_populates="kpi_values")

class Alert(Base):
    """System alerts and notifications"""
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"))
    title = Column(String)
    description = Column(String)
    priority = Column(String)  # high, medium, low
    category = Column(String)  # e.g., "water_quality", "infrastructure"
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="alerts")

class Report(Base):
    """Saved reports and dashboards"""
    __tablename__ = "reports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"))
    name = Column(String)
    description = Column(String, nullable=True)
    content = Column(Text)  # JSON config of report
    created_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="reports")

class FinancialModel(Base):
    """Financial impact models"""
    __tablename__ = "financial_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"))
    name = Column(String)
    description = Column(String, nullable=True)
    parameters = Column(Text)  # JSON string of model parameters
    created_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow)

class DataQualityCheck(Base):
    """Data quality check results"""
    __tablename__ = "data_quality_checks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"))
    dataset_id = Column(String, ForeignKey("datasets.id"))
    check_type = Column(String)  # e.g., "completeness", "accuracy"
    status = Column(String)  # pass, fail, warning
    details = Column(Text)  # JSON details of check results
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    """Audit log for system activities"""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String)
    resource_type = Column(String)  # e.g., "user", "report", "dataset"
    resource_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
