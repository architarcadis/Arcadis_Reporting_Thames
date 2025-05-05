# Client management routes for the API

from fastapi import APIRouter, Depends, HTTPException, status, Response, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import json
import uuid
import os
import logging
from pydantic import BaseModel, Field
from datetime import datetime

from .. import models, database, auth
from ..auth import get_current_user, get_current_admin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the router
router = APIRouter()

# --- Models for request/response validation ---

class TenantBase(BaseModel):
    """Base model for tenant data"""
    name: str
    subscription_tier: Optional[str] = "standard"
    settings: Optional[Dict[str, Any]] = None

class TenantCreate(TenantBase):
    """Model for creating a new tenant"""
    pass

class TenantUpdate(BaseModel):
    """Model for updating an existing tenant"""
    name: Optional[str] = None
    subscription_tier: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class Tenant(TenantBase):
    """Model for tenant response data"""
    id: str
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class UserBase(BaseModel):
    """Base model for user data"""
    username: str
    email: str
    is_admin: Optional[bool] = False

class UserCreate(UserBase):
    """Model for creating a new user"""
    password: str

class UserUpdate(BaseModel):
    """Model for updating an existing user"""
    email: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None

class User(UserBase):
    """Model for user response data"""
    id: str
    tenant_id: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class ThemeUpdate(BaseModel):
    """Model for updating tenant theme"""
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    success_color: Optional[str] = None
    warning_color: Optional[str] = None
    danger_color: Optional[str] = None
    logo_url: Optional[str] = None

# --- Routes ---

@router.get("/", response_model=List[Tenant], tags=["Tenants"])
async def get_all_tenants(
    skip: int = 0, 
    limit: int = 100,
    active_only: bool = True,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_admin)
):
    """
    Get all tenants (admin only)
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        active_only: Return only active tenants
        
    Returns:
        List of tenants
    """
    query = db.query(models.Tenant)
    
    if active_only:
        query = query.filter(models.Tenant.is_active == True)
        
    tenants = query.offset(skip).limit(limit).all()
    return tenants

@router.post("/", response_model=Tenant, status_code=status.HTTP_201_CREATED, tags=["Tenants"])
async def create_tenant(
    tenant: TenantCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_admin)
):
    """
    Create a new tenant (admin only)
    
    Args:
        tenant: Tenant data
        
    Returns:
        Created tenant
    """
    # Check if tenant with same name already exists
    existing_tenant = db.query(models.Tenant).filter(models.Tenant.name == tenant.name).first()
    if existing_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant with this name already exists"
        )
    
    # Convert settings dict to JSON string if provided
    settings_json = json.dumps(tenant.settings) if tenant.settings else json.dumps({
        "primary_color": "#005670",
        "secondary_color": "#00A1D6",
        "success_color": "#28A745",
        "warning_color": "#FFB107",
        "danger_color": "#FF4B4B",
        "logo_url": None
    })
    
    # Create new tenant
    new_tenant = models.Tenant(
        id=str(uuid.uuid4()),
        name=tenant.name,
        subscription_tier=tenant.subscription_tier,
        settings=settings_json,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)
    
    logger.info(f"Created new tenant: {new_tenant.name} (ID: {new_tenant.id})")
    return new_tenant

@router.get("/{tenant_id}", response_model=Tenant, tags=["Tenants"])
async def get_tenant(
    tenant_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get tenant details
    
    Args:
        tenant_id: ID of the tenant
        
    Returns:
        Tenant details
    """
    # Regular users can only access their own tenant
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this tenant"
        )
    
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
        
    return tenant

@router.put("/{tenant_id}", response_model=Tenant, tags=["Tenants"])
async def update_tenant(
    tenant_id: str,
    tenant_update: TenantUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_admin)
):
    """
    Update tenant details (admin only)
    
    Args:
        tenant_id: ID of the tenant to update
        tenant_update: Updated tenant data
        
    Returns:
        Updated tenant
    """
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    # Update fields if provided
    if tenant_update.name is not None:
        # Check if the new name is already taken by another tenant
        existing = db.query(models.Tenant).filter(
            models.Tenant.name == tenant_update.name,
            models.Tenant.id != tenant_id
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant with this name already exists"
            )
            
        tenant.name = tenant_update.name
        
    if tenant_update.subscription_tier is not None:
        tenant.subscription_tier = tenant_update.subscription_tier
        
    if tenant_update.settings is not None:
        # Update settings - merge with existing
        current_settings = json.loads(tenant.settings) if tenant.settings else {}
        current_settings.update(tenant_update.settings)
        tenant.settings = json.dumps(current_settings)
        
    if tenant_update.is_active is not None:
        tenant.is_active = tenant_update.is_active
    
    db.commit()
    db.refresh(tenant)
    
    logger.info(f"Updated tenant: {tenant.name} (ID: {tenant.id})")
    return tenant

@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Tenants"])
async def delete_tenant(
    tenant_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_admin)
):
    """
    Delete or deactivate a tenant (admin only)
    
    Args:
        tenant_id: ID of the tenant to delete
    """
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    # Instead of actually deleting, just deactivate
    tenant.is_active = False
    db.commit()
    
    logger.info(f"Deactivated tenant: {tenant.name} (ID: {tenant.id})")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.put("/{tenant_id}/theme", tags=["Tenants"])
async def update_tenant_theme(
    tenant_id: str,
    theme: ThemeUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Update tenant theme settings
    
    Args:
        tenant_id: ID of the tenant
        theme: Updated theme settings
        
    Returns:
        Updated theme settings
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this tenant's theme"
        )
    
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    # Get current settings
    current_settings = json.loads(tenant.settings) if tenant.settings else {}
    
    # Update theme settings
    if theme.primary_color is not None:
        current_settings["primary_color"] = theme.primary_color
        
    if theme.secondary_color is not None:
        current_settings["secondary_color"] = theme.secondary_color
        
    if theme.success_color is not None:
        current_settings["success_color"] = theme.success_color
        
    if theme.warning_color is not None:
        current_settings["warning_color"] = theme.warning_color
        
    if theme.danger_color is not None:
        current_settings["danger_color"] = theme.danger_color
        
    if theme.logo_url is not None:
        current_settings["logo_url"] = theme.logo_url
    
    # Save updated settings
    tenant.settings = json.dumps(current_settings)
    db.commit()
    
    logger.info(f"Updated theme for tenant: {tenant.name} (ID: {tenant.id})")
    return current_settings

@router.get("/{tenant_id}/theme", tags=["Tenants"])
async def get_tenant_theme(
    tenant_id: str,
    db: Session = Depends(database.get_db)
):
    """
    Get tenant theme settings
    
    Args:
        tenant_id: ID of the tenant
        
    Returns:
        Theme settings
    """
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
    
    # Parse settings
    settings = json.loads(tenant.settings) if tenant.settings else {}
    
    # Ensure all theme properties exist
    theme = {
        "primary_color": settings.get("primary_color", "#005670"),
        "secondary_color": settings.get("secondary_color", "#00A1D6"),
        "success_color": settings.get("success_color", "#28A745"),
        "warning_color": settings.get("warning_color", "#FFB107"),
        "danger_color": settings.get("danger_color", "#FF4B4B"),
        "logo_url": settings.get("logo_url")
    }
    
    return theme

# --- User management routes ---

@router.get("/{tenant_id}/users", response_model=List[User], tags=["Users"])
async def get_tenant_users(
    tenant_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get all users for a tenant
    
    Args:
        tenant_id: ID of the tenant
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of users
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this tenant's users"
        )
    
    users = db.query(models.User).filter(models.User.tenant_id == tenant_id).offset(skip).limit(limit).all()
    return users

@router.post("/{tenant_id}/users", response_model=User, status_code=status.HTTP_201_CREATED, tags=["Users"])
async def create_tenant_user(
    tenant_id: str,
    user: UserCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Create a new user for a tenant
    
    Args:
        tenant_id: ID of the tenant
        user: User data
        
    Returns:
        Created user
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create users for this tenant"
        )
    
    # Verify tenant exists
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
        
    # Check if username already exists
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
        
    # Check if email already exists
    existing_email = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    
    # Create new user
    hashed_password = auth.get_password_hash(user.password)
    
    new_user = models.User(
        id=str(uuid.uuid4()),
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        is_active=True,
        is_admin=user.is_admin if current_user.is_admin else False,  # Only admins can create admin users
        tenant_id=tenant_id,
        created_at=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"Created new user: {new_user.username} for tenant: {tenant.name}")
    return new_user

@router.get("/{tenant_id}/users/{user_id}", response_model=User, tags=["Users"])
async def get_tenant_user(
    tenant_id: str,
    user_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get user details
    
    Args:
        tenant_id: ID of the tenant
        user_id: ID of the user
        
    Returns:
        User details
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user"
        )
    
    user = db.query(models.User).filter(
        models.User.id == user_id,
        models.User.tenant_id == tenant_id
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
        
    return user

@router.put("/{tenant_id}/users/{user_id}", response_model=User, tags=["Users"])
async def update_tenant_user(
    tenant_id: str,
    user_id: str,
    user_update: UserUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Update user details
    
    Args:
        tenant_id: ID of the tenant
        user_id: ID of the user
        user_update: Updated user data
        
    Returns:
        Updated user
    """
    # Check permissions
    is_self_update = current_user.id == user_id
    is_admin_for_tenant = current_user.is_admin and (current_user.tenant_id == tenant_id or current_user.tenant_id is None)
    
    if not is_self_update and not is_admin_for_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user"
        )
    
    user = db.query(models.User).filter(
        models.User.id == user_id,
        models.User.tenant_id == tenant_id
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update fields if provided
    if user_update.email is not None:
        # Check if email already exists for another user
        existing_email = db.query(models.User).filter(
            models.User.email == user_update.email,
            models.User.id != user_id
        ).first()
        
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
            
        user.email = user_update.email
    
    if user_update.password is not None:
        user.hashed_password = auth.get_password_hash(user_update.password)
    
    # Only admins can update these fields
    if is_admin_for_tenant:
        if user_update.is_active is not None:
            user.is_active = user_update.is_active
            
        if user_update.is_admin is not None:
            user.is_admin = user_update.is_admin
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"Updated user: {user.username} (ID: {user.id})")
    return user

@router.delete("/{tenant_id}/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Users"])
async def delete_tenant_user(
    tenant_id: str,
    user_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Delete or deactivate a user
    
    Args:
        tenant_id: ID of the tenant
        user_id: ID of the user
    """
    # Check permissions
    is_admin_for_tenant = current_user.is_admin and (current_user.tenant_id == tenant_id or current_user.tenant_id is None)
    
    if not is_admin_for_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete users for this tenant"
        )
    
    user = db.query(models.User).filter(
        models.User.id == user_id,
        models.User.tenant_id == tenant_id
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent deletion of the last admin user for a tenant
    if user.is_admin:
        admin_count = db.query(models.User).filter(
            models.User.tenant_id == tenant_id,
            models.User.is_admin == True,
            models.User.is_active == True
        ).count()
        
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the last admin user for this tenant"
            )
    
    # Instead of actually deleting, just deactivate
    user.is_active = False
    db.commit()
    
    logger.info(f"Deactivated user: {user.username} (ID: {user.id})")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
