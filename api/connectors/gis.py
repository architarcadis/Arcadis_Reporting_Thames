# GIS (Geographic Information System) connector implementation

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import requests
import json
import os
import logging
from datetime import datetime, timedelta
import geopandas as gpd  # For geospatial data
import shapely.wkt  # For WKT parsing

from .base import BaseConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GisConnector(BaseConnector):
    """
    Connector for GIS (Geographic Information System) platforms
    
    Supports:
    - ESRI ArcGIS REST API
    - GeoServer WFS/WMS
    - QGIS Server
    - Direct database connections to PostGIS/SQL Server Spatial
    """
    
    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize GIS connector
        
        Args:
            connection_params: Dictionary containing:
                - gis_type: GIS system type ('arcgis', 'geoserver', 'qgis', 'postgis', 'sql_spatial')
                - base_url: Base URL for the GIS service (for API-based systems)
                - username: Authentication username (if required)
                - password: Authentication password (if required)
                - api_key: API key (if applicable)
                - token_url: URL for obtaining authentication token (if applicable)
                - host: Database host (for DB-based systems)
                - port: Database port (for DB-based systems)
                - database: Database name (for DB-based systems)
        """
        super().__init__(connection_params)
        self.gis_type = connection_params.get('gis_type', 'arcgis').lower()
        self.token = None
        self.token_expiry = None
    
    def connect(self) -> bool:
        """
        Establish connection to the GIS system
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.gis_type in ['arcgis', 'geoserver', 'qgis']:
                # For API-based GIS systems, authenticate and get token if needed
                if self.connection_params.get('username') and self.connection_params.get('password'):
                    self._authenticate()
                else:
                    # No authentication needed, just verify the service is accessible
                    test_url = f"{self.connection_params.get('base_url')}"
                    if self.gis_type == 'arcgis':
                        test_url += "/info?f=json"
                    elif self.gis_type == 'geoserver':
                        test_url += "/ows?service=WFS&version=1.0.0&request=GetCapabilities"
                    
                    response = requests.get(test_url, timeout=30)
                    if response.status_code != 200:
                        logger.error(f"Failed to connect to GIS service: {response.status_code}")
                        return False
                
                self.connection = {
                    'base_url': self.connection_params.get('base_url'),
                    'token': self.token
                }
                self.connected = True
                self.log_activity("connect", {"gis_type": self.gis_type})
                return True
                
            elif self.gis_type in ['postgis', 'sql_spatial']:
                # Connect to spatial database
                if self.gis_type == 'postgis':
                    # Connection string for PostGIS
                    from sqlalchemy import create_engine
                    conn_str = (
                        f"postgresql://{self.connection_params.get('username')}:"
                        f"{self.connection_params.get('password')}@"
                        f"{self.connection_params.get('host')}:"
                        f"{self.connection_params.get('port')}/"
                        f"{self.connection_params.get('database')}"
                    )
                    self.connection = create_engine(conn_str)
                    
                elif self.gis_type == 'sql_spatial':
                    # Connection string for SQL Server Spatial
                    import pyodbc
                    conn_str = (
                        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                        f"SERVER={self.connection_params.get('host')};"
                        f"DATABASE={self.connection_params.get('database')};"
                        f"UID={self.connection_params.get('username')};"
                        f"PWD={self.connection_params.get('password')}"
                    )
                    self.connection = pyodbc.connect(conn_str)
                
                self.connected = True
                self.log_activity("connect", {"gis_type": self.gis_type, "database": self.connection_params.get('database')})
                return True
                
            else:
                logger.error(f"Unsupported GIS type: {self.gis_type}")
                self.last_error = f"Unsupported GIS type: {self.gis_type}"
                return False
                
        except Exception as e:
            self.handle_error(e, "connect")
            return False
    
    def _authenticate(self) -> bool:
        """
        Authenticate with the GIS service and get token
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            if self.gis_type == 'arcgis':
                # ArcGIS authentication
                token_url = self.connection_params.get('token_url') or f"{self.connection_params.get('base_url')}/generateToken"
                
                params = {
                    'username': self.connection_params.get('username'),
                    'password': self.connection_params.get('password'),
                    'client': 'referer',
                    'referer': self.connection_params.get('referer', 'http://localhost'),
                    'expiration': 60,  # Token valid for 60 minutes
                    'f': 'json'
                }
                
                response = requests.post(token_url, data=params, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'token' in result:
                        self.token = result['token']
                        self.token_expiry = datetime.now() + timedelta(minutes=55)  # Refresh before it expires
                        return True
                    else:
                        logger.error(f"Authentication failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    logger.error(f"Authentication request failed: {response.status_code}")
                    return False
                    
            elif self.gis_type == 'geoserver':
                # GeoServer basic auth - store credentials for requests
                self.token = {
                    'auth': (
                        self.connection_params.get('username'),
                        self.connection_params.get('password')
                    )
                }
                return True
                
            return False
            
        except Exception as e:
            self.handle_error(e, "_authenticate")
            return False
    
    def disconnect(self) -> bool:
        """
        Close connection to the GIS system
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.gis_type in ['postgis', 'sql_spatial'] and self.connection:
                if self.gis_type == 'postgis':
                    # No explicit disconnect for SQLAlchemy engines
                    pass
                elif self.gis_type == 'sql_spatial':
                    self.connection.close()
            
            self.connected = False
            self.token = None
            self.token_expiry = None
            self.log_activity("disconnect", {"gis_type": self.gis_type})
            return True
            
        except Exception as e:
            self.handle_error(e, "disconnect")
            return False
    
    def test_connection(self) -> bool:
        """
        Test if the connection to the GIS system is working
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            if not self.connected:
                return self.connect()
                
            if self.gis_type == 'arcgis':
                # Test ArcGIS connection
                test_url = f"{self.connection['base_url']}/info"
                params = {'f': 'json'}
                
                if self.token:
                    params['token'] = self.token
                
                response = requests.get(test_url, params=params, timeout=10)
                return response.status_code == 200
                
            elif self.gis_type == 'geoserver':
                # Test GeoServer connection
                test_url = f"{self.connection['base_url']}/ows?service=WFS&version=1.0.0&request=GetCapabilities"
                
                if self.token:
                    response = requests.get(test_url, auth=self.token['auth'], timeout=10)
                else:
                    response = requests.get(test_url, timeout=10)
                    
                return response.status_code == 200
                
            elif self.gis_type in ['postgis', 'sql_spatial']:
                # Test database connection
                if self.gis_type == 'postgis':
                    try:
                        with self.connection.connect() as conn:
                            result = conn.execute("SELECT PostGIS_Version()").fetchone()
                        return True
                    except:
                        return False
                
                elif self.gis_type == 'sql_spatial':
                    try:
                        cursor = self.connection.cursor()
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        cursor.close()
                        return result[0] == 1
                    except:
                        return False
            
            return self.connected
            
        except Exception as e:
            self.handle_error(e, "test_connection")
            return False
    
    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from the GIS system
        
        Args:
            query: For API-based: layer/feature class name or endpoint
                  For DB-based: SQL query string
            params: Optional parameters:
                - where_clause: SQL WHERE clause for filtering (API-based)
                - out_fields: Fields to return (API-based)
                - geometry: Spatial filter geometry
                - geometry_type: Type of geometry filter
                - spatial_rel: Spatial relationship
                - return_geometry: Whether to return geometries
                
        Returns:
            GeoDataFrame containing the fetched data
        """
        if not self.connected and not self.connect():
            return pd.DataFrame()
            
        try:
            params = params or {}
            
            if self.gis_type == 'arcgis':
                # Fetch data from ArcGIS REST API
                layer_url = f"{self.connection['base_url']}/{query}/query"
                
                request_params = {
                    'f': 'json',
                    'outFields': params.get('out_fields', '*'),
                    'where': params.get('where_clause', '1=1'),
                    'returnGeometry': params.get('return_geometry', True),
                    'outSR': params.get('out_sr', '4326')  # Default to WGS84
                }
                
                if self.token:
                    request_params['token'] = self.token
                
                # Add spatial filter if provided
                if 'geometry' in params:
                    request_params['geometry'] = params['geometry']
                    request_params['geometryType'] = params.get('geometry_type', 'esriGeometryEnvelope')
                    request_params['spatialRel'] = params.get('spatial_rel', 'esriSpatialRelIntersects')
                
                response = requests.get(layer_url, params=request_params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'features' not in data or len(data['features']) == 0:
                        return gpd.GeoDataFrame()
                    
                    # Extract attributes and geometries
                    features = []
                    for feature in data['features']:
                        # Get attributes
                        attributes = feature['attributes']
                        
                        # Get geometry if available
                        geometry = None
                        if 'geometry' in feature and params.get('return_geometry', True):
                            # Convert ESRI JSON geometry to Shapely geometry
                            geom_json = feature['geometry']
                            if 'x' in geom_json and 'y' in geom_json:  # Point
                                from shapely.geometry import Point
                                geometry = Point(geom_json['x'], geom_json['y'])
                            elif 'rings' in geom_json:  # Polygon
                                from shapely.geometry import Polygon, MultiPolygon
                                rings = geom_json['rings']
                                polygons = [Polygon(ring) for ring in rings]
                                if len(polygons) == 1:
                                    geometry = polygons[0]
                                else:
                                    geometry = MultiPolygon(polygons)
                            elif 'paths' in geom_json:  # Line
                                from shapely.geometry import LineString, MultiLineString
                                paths = geom_json['paths']
                                lines = [LineString(path) for path in paths]
                                if len(lines) == 1:
                                    geometry = lines[0]
                                else:
                                    geometry = MultiLineString(lines)
                        
                        # Add geometry to attributes
                        attributes['geometry'] = geometry
                        features.append(attributes)
                    
                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(features, geometry='geometry', crs="EPSG:4326")
                    return gdf
                else:
                    logger.error(f"ArcGIS API error: {response.status_code} - {response.text}")
                    return gpd.GeoDataFrame()
                    
            elif self.gis_type == 'geoserver':
                # Fetch data from GeoServer WFS
                wfs_url = f"{self.connection['base_url']}/ows"
                
                request_params = {
                    'service': 'WFS',
                    'version': '2.0.0',
                    'request': 'GetFeature',
                    'typeName': query,
                    'outputFormat': 'application/json'
                }
                
                if 'where_clause' in params:
                    request_params['CQL_FILTER'] = params['where_clause']
                
                if 'max_features' in params:
                    request_params['count'] = params['max_features']
                
                if self.token:
                    response = requests.get(wfs_url, params=request_params, auth=self.token['auth'], timeout=60)
                else:
                    response = requests.get(wfs_url, params=request_params, timeout=60)
                
                if response.status_code == 200:
                    # Parse GeoJSON response into GeoDataFrame
                    gdf = gpd.GeoDataFrame.from_features(response.json(), crs="EPSG:4326")
                    return gdf
                else:
                    logger.error(f"GeoServer WFS error: {response.status_code} - {response.text}")
                    return gpd.GeoDataFrame()
                    
            elif self.gis_type in ['postgis', 'sql_spatial']:
                # Fetch data from spatial database
                if self.gis_type == 'postgis':
                    # For PostGIS, use GeoPandas read_postgis
                    gdf = gpd.read_postgis(
                        query, 
                        self.connection, 
                        geom_col=params.get('geometry_column', 'geom'),
                        crs=params.get('crs', "EPSG:4326")
                    )
                    return gdf
                
                elif self.gis_type == 'sql_spatial':
                    # For SQL Server Spatial, execute query and convert WKT to geometries
                    df = pd.read_sql(query, self.connection)
                    
                    # Check if the geometry column exists
                    geom_col = params.get('geometry_column', 'Shape')
                    if geom_col in df.columns:
                        # Convert WKT strings to Shapely geometries
                        df['geometry'] = df[geom_col].apply(lambda x: shapely.wkt.loads(x) if x else None)
                        # Create GeoDataFrame
                        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=params.get('crs', "EPSG:4326"))
                        return gdf
                    else:
                        return gpd.GeoDataFrame(df, crs=params.get('crs', "EPSG:4326"))
            
            # Return empty DataFrame if GIS type not implemented
            logger.warning(f"Fetch not fully implemented for GIS type: {self.gis_type}")
            return gpd.GeoDataFrame()
            
        except Exception as e:
            self.handle_error(e, "fetch_data")
            return gpd.GeoDataFrame()
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of available datasets from the GIS system
        
        Returns:
            List of dictionaries with dataset metadata including:
            - id: Dataset identifier
            - name: Human-readable name
            - description: Description if available
            - type: Feature type (point, line, polygon)
            - layer_type: Layer type (feature layer, raster, etc.)
            - fields: List of available fields
        """
        if not self.connected and not self.connect():
            return []
            
        try:
            if self.gis_type == 'arcgis':
                # Get services info from ArcGIS REST API
                services_url = f"{self.connection['base_url']}"
                params = {'f': 'json'}
                
                if self.token:
                    params['token'] = self.token
                
                response = requests.get(services_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    layers = []
                    
                    # Process layers list
                    if 'layers' in data:
                        for layer in data['layers']:
                            layers.append({
                                'id': layer['id'],
                                'name': layer['name'],
                                'description': layer.get('description', ''),
                                'type': self._get_geometry_type(layer.get('geometryType', '')),
                                'layer_type': 'Feature Layer'
                            })
                    
                    # Process tables list
                    if 'tables' in data:
                        for table in data['tables']:
                            layers.append({
                                'id': table['id'],
                                'name': table['name'],
                                'description': table.get('description', ''),
                                'type': 'Table',
                                'layer_type': 'Table'
                            })
                    
                    return layers
                else:
                    logger.error(f"ArcGIS API error: {response.status_code} - {response.text}")
                    return []
                    
            elif self.gis_type == 'geoserver':
                # Get available layers from GeoServer
                wfs_url = f"{self.connection['base_url']}/ows"
                params = {
                    'service': 'WFS',
                    'version': '2.0.0',
                    'request': 'GetCapabilities'
                }
                
                if self.token:
                    response = requests.get(wfs_url, params=params, auth=self.token['auth'], timeout=30)
                else:
                    response = requests.get(wfs_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    # Parse XML response - simplified example
                    import xml.etree.ElementTree as ET
                    
                    # Parse XML
                    root = ET.fromstring(response.content)
                    
                    # Define XML namespaces
                    namespaces = {
                        'wfs': 'http://www.opengis.net/wfs/2.0',
                        'fes': 'http://www.opengis.net/fes/2.0',
                        'ows': 'http://www.opengis.net/ows/1.1'
                    }
                    
                    # Find feature types
                    feature_types = root.findall('.//wfs:FeatureType', namespaces)
                    
                    layers = []
                    for ft in feature_types:
                        name = ft.find('wfs:Name', namespaces)
                        title = ft.find('wfs:Title', namespaces)
                        abstract = ft.find('wfs:Abstract', namespaces)
                        
                        layers.append({
                            'id': name.text if name is not None else '',
                            'name': title.text if title is not None else name.text if name is not None else '',
                            'description': abstract.text if abstract is not None else '',
                            'type': 'Unknown',  # WFS capabilities doesn't include geometry type
                            'layer_type': 'Feature Layer'
                        })
                    
                    return layers
                else:
                    logger.error(f"GeoServer WFS error: {response.status_code} - {response.text}")
                    return []
                    
            elif self.gis_type in ['postgis', 'sql_spatial']:
                # Get available spatial tables from database
                if self.gis_type == 'postgis':
                    # Query for PostGIS spatial tables
                    query = """
                    SELECT
                        f_table_schema as schema,
                        f_table_name as table_name,
                        type as geometry_type,
                        srid
                    FROM geometry_columns
                    ORDER BY f_table_schema, f_table_name
                    """
                    
                    df = pd.read_sql(query, self.connection)
                    
                    layers = []
                    for _, row in df.iterrows():
                        layers.append({
                            'id': f"{row['schema']}.{row['table_name']}",
                            'name': row['table_name'],
                            'description': f"Geometry type: {row['geometry_type']}, SRID: {row['srid']}",
                            'type': self._get_geometry_type(row['geometry_type']),
                            'layer_type': 'Feature Layer'
                        })
                    
                    return layers
                
                elif self.gis_type == 'sql_spatial':
                    # Query for SQL Server spatial tables
                    query = """
                    SELECT
                        SCHEMA_NAME(t.schema_id) as schema_name,
                        t.name as table_name,
                        c.name as geometry_column,
                        c.system_type_id
                    FROM sys.tables t
                    JOIN sys.columns c ON t.object_id = c.object_id
                    WHERE c.system_type_id IN (240, 241)  -- Geometry or Geography type
                    ORDER BY schema_name, table_name
                    """
                    
                    cursor = self.connection.cursor()
                    cursor.execute(query)
                    
                    columns = [column[0] for column in cursor.description]
                    results = cursor.fetchall()
                    df = pd.DataFrame.from_records(results, columns=columns)
                    
                    layers = []
                    for _, row in df.iterrows():
                        geometry_type = 'Geometry' if row['system_type_id'] == 240 else 'Geography'
                        layers.append({
                            'id': f"{row['schema_name']}.{row['table_name']}",
                            'name': row['table_name'],
                            'description': f"Geometry column: {row['geometry_column']}, Type: {geometry_type}",
                            'type': 'Unknown',  # Need additional query to determine actual geometry type
                            'layer_type': 'Feature Layer'
                        })
                    
                    return layers
            
            # Return empty list if GIS type not implemented
            logger.warning(f"Getting available datasets not fully implemented for GIS type: {self.gis_type}")
            return []
            
        except Exception as e:
            self.handle_error(e, "get_available_datasets")
            return []
    
    def _get_geometry_type(self, esri_type: str) -> str:
        """
        Convert ESRI geometry type to standard type
        
        Args:
            esri_type: ESRI geometry type string
            
        Returns:
            Standardized geometry type string
        """
        esri_types = {
            'esriGeometryPoint': 'Point',
            'esriGeometryMultipoint': 'MultiPoint',
            'esriGeometryPolyline': 'LineString',
            'esriGeometryPolygon': 'Polygon',
            'POINT': 'Point',
            'MULTIPOINT': 'MultiPoint',
            'LINESTRING': 'LineString',
            'MULTILINESTRING': 'MultiLineString',
            'POLYGON': 'Polygon',
            'MULTIPOLYGON': 'MultiPolygon'
        }
        
        return esri_types.get(esri_type, 'Unknown')
