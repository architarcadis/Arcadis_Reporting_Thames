# Data ingestion routes for the API

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import json
import uuid
import os
import pandas as pd
import logging
from pydantic import BaseModel, Field
from datetime import datetime
import traceback
import io

from .. import models, database, auth, transformers, validators
from ..auth import get_current_user
from ..connectors import scada, gis, billing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the router
router = APIRouter()

# --- Models for request/response validation ---

class DataSourceBase(BaseModel):
    """Base model for data source"""
    name: str
    type: str  # 'scada', 'billing', 'gis', 'file', 'api'
    connection_string: Optional[str] = None
    api_key: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

class DataSourceCreate(DataSourceBase):
    """Model for creating a new data source"""
    pass

class DataSourceUpdate(BaseModel):
    """Model for updating an existing data source"""
    name: Optional[str] = None
    connection_string: Optional[str] = None
    api_key: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class DataSource(DataSourceBase):
    """Model for data source response"""
    id: str
    tenant_id: str
    is_active: bool
    last_sync: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        orm_mode = True

class FileUploadResponse(BaseModel):
    """Model for file upload response"""
    file_id: str
    filename: str
    size: int
    data_type: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: Optional[List[str]] = None
    validation_result: Optional[Dict[str, Any]] = None

class ConnectorTestRequest(BaseModel):
    """Model for testing a connector"""
    connection_params: Dict[str, Any]
    connector_type: str  # 'scada', 'billing', 'gis'

class SyncRequest(BaseModel):
    """Model for data synchronization request"""
    datasource_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    entity_type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class SyncStatus(BaseModel):
    """Model for sync status response"""
    sync_id: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    start_time: datetime
    end_time: Optional[datetime] = None
    row_count: Optional[int] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class DatasetInfo(BaseModel):
    """Model for dataset information"""
    id: str
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    update_frequency: Optional[str] = None

# --- Helper functions ---

def process_file_upload(file_content: bytes, filename: str, tenant_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an uploaded file
    
    Args:
        file_content: Content of the uploaded file
        filename: Name of the uploaded file
        tenant_id: ID of the tenant
        settings: Settings for file processing
        
    Returns:
        Dictionary with information about the processed file
    """
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Determine file type and read data
    try:
        if file_extension == '.csv':
            # Read CSV file
            delimiter = settings.get('delimiter', ',')
            df = pd.read_csv(io.BytesIO(file_content), delimiter=delimiter, encoding=settings.get('encoding', 'utf-8'))
        elif file_extension in ['.xls', '.xlsx']:
            # Read Excel file
            sheet_name = settings.get('sheet_name', 0)
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
        elif file_extension == '.json':
            # Read JSON file
            df = pd.read_json(io.BytesIO(file_content), orient=settings.get('orient', 'records'))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Apply transformations if specified
        if settings.get('apply_transformations', False):
            client_format = settings.get('client_format', 'generic')
            df = transformers.transform_data(df, client_format=client_format)
        
        # Validate data if specified
        validation_result = None
        if settings.get('validate_data', False):
            data_type = settings.get('data_type', 'generic')
            validation_result = validators.validate_data(df, data_type=data_type)
        
        # Store metadata about the imported data
        result = {
            'file_id': file_id,
            'filename': filename,
            'size': len(file_content),
            'data_type': settings.get('data_type', 'generic'),
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'validation_result': validation_result
        }
        
        # TODO: Store the actual data in database or file system
        # For now, we'll just return the metadata
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing file: {str(e)}"
        )

def run_data_sync(datasource_id: str, sync_id: str, tenant_id: str, db: Session, params: Dict[str, Any] = None):
    """
    Run data synchronization as a background task
    
    Args:
        datasource_id: ID of the data source
        sync_id: ID of the sync operation
        tenant_id: ID of the tenant
        db: Database session
        params: Sync parameters
    """
    logger.info(f"Starting data sync {sync_id} for datasource {datasource_id}")
    
    try:
        # Create sync record
        sync_record = models.DataSync(
            id=sync_id,
            datasource_id=datasource_id,
            status="in_progress",
            start_time=datetime.utcnow()
        )
        db.add(sync_record)
        db.commit()
        
        # Get data source
        datasource = db.query(models.DataSource).filter(
            models.DataSource.id == datasource_id,
            models.DataSource.tenant_id == tenant_id
        ).first()
        
        if not datasource:
            raise ValueError(f"Data source {datasource_id} not found")
        
        # Initialize appropriate connector
        settings = json.loads(datasource.settings) if datasource.settings else {}
        connection_params = {
            'host': datasource.connection_string,
            'api_key': datasource.api_key,
            **settings
        }
        
        connector = None
        if datasource.type == 'scada':
            connector = scada.ScadaConnector(connection_params)
        elif datasource.type == 'gis':
            connector = gis.GisConnector(connection_params)
        elif datasource.type == 'billing':
            connector = billing.BillingConnector(connection_params)
        else:
            raise ValueError(f"Unsupported connector type: {datasource.type}")
        
        # Connect to data source
        if not connector.connect():
            raise ValueError(f"Failed to connect to {datasource.type} data source: {connector.last_error}")
        
        # Fetch data based on parameters
        entity_type = params.get('entity_type', '')
        start_date = params.get('start_date', '')
        end_date = params.get('end_date', '')
        
        query_params = {
            'start_date': start_date,
            'end_date': end_date,
            'limit': params.get('limit', 1000)
        }
        
        if datasource.type == 'billing':
            if entity_type == 'customer':
                data = connector.get_customer_data(limit=query_params['limit'])
            elif entity_type == 'consumption':
                data = connector.get_consumption_data(
                    start_date=start_date,
                    end_date=end_date,
                    limit=query_params['limit']
                )
            elif entity_type == 'billing':
                data = connector.get_billing_data(
                    start_date=start_date,
                    end_date=end_date,
                    limit=query_params['limit']
                )
            elif entity_type == 'payment':
                data = connector.get_payment_data(
                    start_date=start_date,
                    end_date=end_date,
                    limit=query_params['limit']
                )
            else:
                data = connector.fetch_data(entity_type, query_params)
        else:
            data = connector.fetch_data(entity_type, query_params)
        
        # Process data
        if len(data) > 0:
            # Apply transformations if needed
            if params.get('apply_transformations', False):
                client_format = params.get('client_format', 'generic')
                data = transformers.transform_data(data, client_format=client_format)
            
            # Validate data if needed
            validation_result = None
            if params.get('validate_data', False):
                data_type = params.get('data_type', 'generic')
                validation_result = validators.validate_data(data, data_type=data_type)
            
            # Create dataset record
            dataset = models.Dataset(
                id=str(uuid.uuid4()),
                datasource_id=datasource_id,
                name=entity_type or "dataset",
                description=f"Imported from {datasource.name} on {datetime.utcnow().isoformat()}",
                schema=json.dumps({'columns': list(data.columns), 'rows': len(data)}),
                last_updated=datetime.utcnow(),
                record_count=len(data)
            )
            db.add(dataset)
            
            # TODO: Store the actual data in database or file system
            # For now, we just record metadata
            
            # Update data source
            datasource.last_sync = datetime.utcnow()
            
            # Update sync record
            sync_record.status = "completed"
            sync_record.end_time = datetime.utcnow()
            sync_record.row_count = len(data)
            sync_record.details = json.dumps({
                'columns': list(data.columns),
                'validation': validation_result
            })
            
            db.commit()
            
            logger.info(f"Completed data sync {sync_id} with {len(data)} records")
        else:
            # No data found
            sync_record.status = "completed"
            sync_record.end_time = datetime.utcnow()
            sync_record.row_count = 0
            sync_record.details = json.dumps({'message': "No data found"})
            
            db.commit()
            
            logger.info(f"Completed data sync {sync_id} with no records")
    
    except Exception as e:
        logger.error(f"Error in data sync {sync_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update sync record with error
        try:
            sync_record = db.query(models.DataSync).filter(models.DataSync.id == sync_id).first()
            if sync_record:
                sync_record.status = "failed"
                sync_record.end_time = datetime.utcnow()
                sync_record.details = json.dumps({
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating sync record: {str(db_error)}")

# --- Routes ---

@router.post("/datasources", response_model=DataSource, status_code=status.HTTP_201_CREATED, tags=["Data Sources"])
async def create_datasource(
    datasource: DataSourceCreate,
    tenant_id: str = Query(..., description="ID of the tenant"),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Create a new data source
    
    Args:
        datasource: Data source details
        tenant_id: ID of the tenant
        
    Returns:
        Created data source
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create data sources for this tenant"
        )
    
    # Check if tenant exists
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    # Check if data source with same name already exists for this tenant
    existing = db.query(models.DataSource).filter(
        models.DataSource.tenant_id == tenant_id,
        models.DataSource.name == datasource.name
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source with this name already exists for this tenant"
        )
    
    # Create data source
    settings_json = json.dumps(datasource.settings) if datasource.settings else None
    
    new_datasource = models.DataSource(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        name=datasource.name,
        type=datasource.type,
        connection_string=datasource.connection_string,
        api_key=datasource.api_key,
        settings=settings_json,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    db.add(new_datasource)
    db.commit()
    db.refresh(new_datasource)
    
    logger.info(f"Created new data source: {new_datasource.name} (ID: {new_datasource.id}) for tenant: {tenant_id}")
    return new_datasource

@router.get("/datasources", response_model=List[DataSource], tags=["Data Sources"])
async def get_datasources(
    tenant_id: str = Query(..., description="ID of the tenant"),
    skip: int = 0,
    limit: int = 100,
    type_filter: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get all data sources for a tenant
    
    Args:
        tenant_id: ID of the tenant
        skip: Number of records to skip
        limit: Maximum number of records to return
        type_filter: Filter by data source type
        active_only: Return only active data sources
        
    Returns:
        List of data sources
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view data sources for this tenant"
        )
    
    # Build query
    query = db.query(models.DataSource).filter(models.DataSource.tenant_id == tenant_id)
    
    if active_only:
        query = query.filter(models.DataSource.is_active == True)
        
    if type_filter:
        query = query.filter(models.DataSource.type == type_filter)
    
    datasources = query.offset(skip).limit(limit).all()
    return datasources

@router.get("/datasources/{datasource_id}", response_model=DataSource, tags=["Data Sources"])
async def get_datasource(
    datasource_id: str,
    tenant_id: str = Query(..., description="ID of the tenant"),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get data source details
    
    Args:
        datasource_id: ID of the data source
        tenant_id: ID of the tenant
        
    Returns:
        Data source details
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view data sources for this tenant"
        )
    
    datasource = db.query(models.DataSource).filter(
        models.DataSource.id == datasource_id,
        models.DataSource.tenant_id == tenant_id
    ).first()
    
    if not datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found"
        )
        
    return datasource

@router.put("/datasources/{datasource_id}", response_model=DataSource, tags=["Data Sources"])
async def update_datasource(
    datasource_id: str,
    datasource_update: DataSourceUpdate,
    tenant_id: str = Query(..., description="ID of the tenant"),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Update data source details
    
    Args:
        datasource_id: ID of the data source
        datasource_update: Updated data source details
        tenant_id: ID of the tenant
        
    Returns:
        Updated data source
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update data sources for this tenant"
        )
    
    datasource = db.query(models.DataSource).filter(
        models.DataSource.id == datasource_id,
        models.DataSource.tenant_id == tenant_id
    ).first()
    
    if not datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found"
        )
    
    # Update fields if provided
    if datasource_update.name is not None:
        # Check if the new name is already taken by another data source for this tenant
        existing = db.query(models.DataSource).filter(
            models.DataSource.tenant_id == tenant_id,
            models.DataSource.name == datasource_update.name,
            models.DataSource.id != datasource_id
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data source with this name already exists for this tenant"
            )
            
        datasource.name = datasource_update.name
        
    if datasource_update.connection_string is not None:
        datasource.connection_string = datasource_update.connection_string
        
    if datasource_update.api_key is not None:
        datasource.api_key = datasource_update.api_key
        
    if datasource_update.settings is not None:
        # Update settings - merge with existing
        current_settings = json.loads(datasource.settings) if datasource.settings else {}
        current_settings.update(datasource_update.settings)
        datasource.settings = json.dumps(current_settings)
        
    if datasource_update.is_active is not None:
        datasource.is_active = datasource_update.is_active
    
    db.commit()
    db.refresh(datasource)
    
    logger.info(f"Updated data source: {datasource.name} (ID: {datasource.id})")
    return datasource

@router.delete("/datasources/{datasource_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Data Sources"])
async def delete_datasource(
    datasource_id: str,
    tenant_id: str = Query(..., description="ID of the tenant"),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Delete or deactivate a data source
    
    Args:
        datasource_id: ID of the data source
        tenant_id: ID of the tenant
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete data sources for this tenant"
        )
    
    datasource = db.query(models.DataSource).filter(
        models.DataSource.id == datasource_id,
        models.DataSource.tenant_id == tenant_id
    ).first()
    
    if not datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found"
        )
    
    # Instead of actually deleting, just deactivate
    datasource.is_active = False
    db.commit()
    
    logger.info(f"Deactivated data source: {datasource.name} (ID: {datasource.id})")
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={})

@router.post("/test-connection", tags=["Data Sources"])
async def test_connection(
    request: ConnectorTestRequest,
    tenant_id: str = Query(..., description="ID of the tenant"),
    current_user: models.User = Depends(get_current_user)
):
    """
    Test connection to a data source
    
    Args:
        request: Connection parameters
        tenant_id: ID of the tenant
        
    Returns:
        Connection test result
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to test connections for this tenant"
        )
    
    try:
        # Initialize appropriate connector
        connector = None
        if request.connector_type.lower() == 'scada':
            connector = scada.ScadaConnector(request.connection_params)
        elif request.connector_type.lower() == 'gis':
            connector = gis.GisConnector(request.connection_params)
        elif request.connector_type.lower() == 'billing':
            connector = billing.BillingConnector(request.connection_params)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported connector type: {request.connector_type}"
            )
        
        # Test connection
        connected = connector.connect()
        if connected:
            connection_test = connector.test_connection()
            connector.disconnect()
            
            if connection_test:
                return {
                    "status": "success",
                    "message": f"Successfully connected to {request.connector_type} data source"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Connected but connection test failed: {connector.last_error}"
                }
        else:
            return {
                "status": "error",
                "message": f"Failed to connect: {connector.last_error}"
            }
    
    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error testing connection: {str(e)}"
        }

@router.post("/upload", response_model=FileUploadResponse, tags=["File Upload"])
async def upload_file(
    file: UploadFile = File(...),
    tenant_id: str = Form(..., description="ID of the tenant"),
    data_type: str = Form("generic", description="Type of data in the file"),
    delimiter: str = Form(",", description="Delimiter for CSV files"),
    sheet_name: Optional[str] = Form(None, description="Sheet name for Excel files"),
    encoding: str = Form("utf-8", description="Encoding for text files"),
    apply_transformations: bool = Form(False, description="Apply transformations to the data"),
    client_format: str = Form("generic", description="Client data format for transformations"),
    validate_data: bool = Form(True, description="Validate the data"),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Upload a data file
    
    Args:
        file: File to upload
        tenant_id: ID of the tenant
        data_type: Type of data in the file
        delimiter: Delimiter for CSV files
        sheet_name: Sheet name for Excel files
        encoding: Encoding for text files
        apply_transformations: Apply transformations to the data
        client_format: Client data format for transformations
        validate_data: Validate the data
        
    Returns:
        Information about the uploaded file
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to upload files for this tenant"
        )
    
    # Check if tenant exists
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    # Check file size (limit to 50MB for this example)
    file_size_limit = 50 * 1024 * 1024  # 50MB
    if file.size > file_size_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {file_size_limit / (1024 * 1024)}MB"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = ['.csv', '.xls', '.xlsx', '.json']
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    file_content = await file.read()
    
    # Process file
    settings = {
        'delimiter': delimiter,
        'sheet_name': sheet_name or 0,
        'encoding': encoding,
        'data_type': data_type,
        'apply_transformations': apply_transformations,
        'client_format': client_format,
        'validate_data': validate_data
    }
    
    result = process_file_upload(file_content, file.filename, tenant_id, settings)
    
    logger.info(f"Processed uploaded file: {file.filename} for tenant: {tenant_id}")
    return result

@router.get("/datasets", response_model=List[DatasetInfo], tags=["Datasets"])
async def get_available_datasets(
    datasource_id: str,
    tenant_id: str = Query(..., description="ID of the tenant"),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get available datasets from a data source
    
    Args:
        datasource_id: ID of the data source
        tenant_id: ID of the tenant
        
    Returns:
        List of available datasets
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access datasets for this tenant"
        )
    
    # Verify data source exists and belongs to tenant
    datasource = db.query(models.DataSource).filter(
        models.DataSource.id == datasource_id,
        models.DataSource.tenant_id == tenant_id,
        models.DataSource.is_active == True
    ).first()
    
    if not datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found or not active"
        )
    
    try:
        # Initialize appropriate connector
        settings = json.loads(datasource.settings) if datasource.settings else {}
        connection_params = {
            'host': datasource.connection_string,
            'api_key': datasource.api_key,
            **settings
        }
        
        connector = None
        if datasource.type == 'scada':
            connector = scada.ScadaConnector(connection_params)
        elif datasource.type == 'gis':
            connector = gis.GisConnector(connection_params)
        elif datasource.type == 'billing':
            connector = billing.BillingConnector(connection_params)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported connector type: {datasource.type}"
            )
        
        # Connect to data source
        if not connector.connect():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to connect to data source: {connector.last_error}"
            )
        
        # Get available datasets
        datasets = connector.get_available_datasets()
        
        # Disconnect
        connector.disconnect()
        
        return datasets
    
    except Exception as e:
        logger.error(f"Error getting available datasets: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting available datasets: {str(e)}"
        )

@router.post("/sync", response_model=SyncStatus, tags=["Data Sync"])
async def sync_data(
    sync_request: SyncRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Query(..., description="ID of the tenant"),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Synchronize data from a data source
    
    Args:
        sync_request: Synchronization parameters
        tenant_id: ID of the tenant
        
    Returns:
        Synchronization status
    """
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to sync data for this tenant"
        )
    
    # Verify data source exists and belongs to tenant
    datasource = db.query(models.DataSource).filter(
        models.DataSource.id == sync_request.datasource_id,
        models.DataSource.tenant_id == tenant_id,
        models.DataSource.is_active == True
    ).first()
    
    if not datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found or not active"
        )
    
    # Create unique ID for this sync operation
    sync_id = str(uuid.uuid4())
    
    # Start sync in background
    params = {
        'entity_type': sync_request.entity_type,
        'start_date': sync_request.start_date,
        'end_date': sync_request.end_date,
        **(sync_request.params or {})
    }
    
    background_tasks.add_task(
        run_data_sync, 
        datasource_id=sync_request.datasource_id, 
        sync_id=sync_id, 
        tenant_id=tenant_id, 
        db=db,
        params=params
    )
    
    # Return initial status
    return SyncStatus(
        sync_id=sync_id,
        status="pending",
        start_time=datetime.utcnow()
    )

@router.get("/sync/{sync_id}", response_model=SyncStatus, tags=["Data Sync"])
async def get_sync_status(
    sync_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get status of a data synchronization operation
    
    Args:
        sync_id: ID of the sync operation
        
    Returns:
        Synchronization status
    """
    # Get sync record
    sync_record = db.query(models.DataSync).filter(models.DataSync.id == sync_id).first()
    
    if not sync_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sync operation not found"
        )
    
    # Get data source to check permission
    datasource = db.query(models.DataSource).filter(models.DataSource.id == sync_record.datasource_id).first()
    
    if not datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated data source not found"
        )
    
    # Check permissions
    if not current_user.is_admin and current_user.tenant_id != datasource.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this sync operation"
        )
    
    # Convert details JSON to dict if present
    details = json.loads(sync_record.details) if sync_record.details else None
    
    return SyncStatus(
        sync_id=sync_record.id,
        status=sync_record.status,
        start_time=sync_record.start_time,
        end_time=sync_record.end_time,
        row_count=sync_record.row_count,
        error_message=details.get('error') if details and 'error' in details else None,
        details=details
    )
