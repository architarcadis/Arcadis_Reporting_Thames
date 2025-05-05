# Billing System connector implementation

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import requests
import json
import os
import logging
from datetime import datetime, timedelta
import time
import pyodbc  # For connecting to billing databases

from .base import BaseConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BillingConnector(BaseConnector):
    """
    Connector for Water Utility Billing Systems
    
    Supports common billing systems used by water utilities:
    - Oracle CC&B (Customer Care & Billing)
    - SAP ISU (Industry Solution for Utilities)
    - Microsoft Dynamics
    - Custom billing APIs
    - SQL-based billing databases
    """
    
    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize Billing connector
        
        Args:
            connection_params: Dictionary containing:
                - system_type: Billing system type ('oracle_ccb', 'sap_isu', 'dynamics', 'custom_api', 'sql')
                - host: Host address
                - port: Port number
                - username: Authentication username
                - password: Authentication password
                - database: Database name (for SQL-based systems)
                - api_key: API key (for API-based systems)
                - service_name: Service name (for Oracle)
                - client: SAP client number (for SAP ISU)
                - api_version: API version (for custom APIs)
        """
        super().__init__(connection_params)
        self.system_type = connection_params.get('system_type', 'sql').lower()
        self.api_base_url = connection_params.get('host')
        
    def connect(self) -> bool:
        """
        Establish connection to the billing system
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.system_type == 'oracle_ccb':
                # Connect to Oracle CC&B database
                import cx_Oracle
                
                dsn = cx_Oracle.makedsn(
                    self.connection_params.get('host'),
                    self.connection_params.get('port'),
                    service_name=self.connection_params.get('service_name')
                )
                
                self.connection = cx_Oracle.connect(
                    user=self.connection_params.get('username'),
                    password=self.connection_params.get('password'),
                    dsn=dsn
                )
                
                self.connected = True
                self.log_activity("connect", {"system": "Oracle CC&B"})
                return True
                
            elif self.system_type == 'sap_isu':
                # Connect to SAP ISU system
                # Note: This would typically use SAP's PyRFC library or similar
                # For this example, we'll simulate the connection
                
                self.connection = {
                    'host': self.connection_params.get('host'),
                    'client': self.connection_params.get('client'),
                    'user': self.connection_params.get('username'),
                    'secured': True
                }
                
                self.connected = True
                self.log_activity("connect", {"system": "SAP ISU", "client": self.connection_params.get('client')})
                return True
                
            elif self.system_type == 'dynamics':
                # Connect to Microsoft Dynamics
                # This would typically use the Dynamics SDK or REST API
                
                auth_url = f"https://{self.connection_params.get('host')}/api/auth"
                response = requests.post(
                    auth_url,
                    json={
                        'username': self.connection_params.get('username'),
                        'password': self.connection_params.get('password')
                    },
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.connection = {
                        'base_url': f"https://{self.connection_params.get('host')}/api",
                        'token': token_data.get('token'),
                        'expires': datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600))
                    }
                    
                    self.connected = True
                    self.log_activity("connect", {"system": "Microsoft Dynamics"})
                    return True
                else:
                    logger.error(f"Failed to authenticate with Dynamics: {response.status_code}")
                    return False
                    
            elif self.system_type == 'custom_api':
                # Connect to custom billing API
                api_url = f"{self.api_base_url}/auth"
                
                # Some APIs use key-based auth, others use tokens
                if self.connection_params.get('api_key'):
                    self.connection = {
                        'base_url': self.api_base_url,
                        'api_key': self.connection_params.get('api_key'),
                        'headers': {
                            'X-API-Key': self.connection_params.get('api_key'),
                            'Content-Type': 'application/json'
                        }
                    }
                    
                    # Test the connection with a simple request
                    test_url = f"{self.api_base_url}/status"
                    response = requests.get(
                        test_url,
                        headers=self.connection['headers'],
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        self.connected = True
                        self.log_activity("connect", {"system": "Custom API (Key-based)"})
                        return True
                    else:
                        logger.error(f"Failed to connect to Custom API: {response.status_code}")
                        return False
                else:
                    # Token-based authentication
                    response = requests.post(
                        api_url,
                        json={
                            'username': self.connection_params.get('username'),
                            'password': self.connection_params.get('password')
                        },
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        token_data = response.json()
                        self.connection = {
                            'base_url': self.api_base_url,
                            'token': token_data.get('token'),
                            'expires': datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600)),
                            'headers': {
                                'Authorization': f"Bearer {token_data.get('token')}",
                                'Content-Type': 'application/json'
                            }
                        }
                        
                        self.connected = True
                        self.log_activity("connect", {"system": "Custom API (Token-based)"})
                        return True
                    else:
                        logger.error(f"Failed to authenticate with Custom API: {response.status_code}")
                        return False
                        
            elif self.system_type == 'sql':
                # Connect to SQL-based billing database
                driver = self.connection_params.get('driver', 'SQL Server')
                
                conn_str = (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={self.connection_params.get('host')};"
                    f"DATABASE={self.connection_params.get('database')};"
                    f"UID={self.connection_params.get('username')};"
                    f"PWD={self.connection_params.get('password')}"
                )
                
                self.connection = pyodbc.connect(conn_str)
                self.connected = True
                self.log_activity("connect", {"system": "SQL Database", "database": self.connection_params.get('database')})
                return True
                
            else:
                logger.error(f"Unsupported billing system type: {self.system_type}")
                self.last_error = f"Unsupported billing system type: {self.system_type}"
                return False
                
        except Exception as e:
            self.handle_error(e, "connect")
            return False
            
    def disconnect(self) -> bool:
        """
        Close connection to the billing system
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.system_type in ['oracle_ccb', 'sql'] and self.connection:
                self.connection.close()
                
            # For API-based systems, there's typically no explicit disconnect
            
            self.connected = False
            self.log_activity("disconnect", {"system_type": self.system_type})
            return True
            
        except Exception as e:
            self.handle_error(e, "disconnect")
            return False
            
    def test_connection(self) -> bool:
        """
        Test if the connection to the billing system is working
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            if not self.connected:
                return self.connect()
                
            if self.system_type == 'oracle_ccb':
                # Test Oracle connection
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                cursor.close()
                return result[0] == 1
                
            elif self.system_type == 'sap_isu':
                # For SAP ISU, we might check connectivity with a simple RFC call
                # This is a simplified example
                return True
                
            elif self.system_type in ['dynamics', 'custom_api']:
                # For API-based systems, check if token is expired
                if self.connection.get('expires') and datetime.now() > self.connection.get('expires'):
                    # Token expired, reconnect
                    return self.connect()
                
                # Test API connectivity with a status endpoint
                test_url = f"{self.connection['base_url']}/status"
                
                headers = self.connection.get('headers', {})
                
                response = requests.get(test_url, headers=headers, timeout=10)
                return response.status_code == 200
                
            elif self.system_type == 'sql':
                # Test SQL connection
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                return result[0] == 1
                
            return self.connected
            
        except Exception as e:
            self.handle_error(e, "test_connection")
            return False
            
    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from the billing system
        
        Args:
            query: For SQL-based: SQL query string
                  For API-based: endpoint path or query name
            params: Optional parameters for the query:
                - start_date: Start date for data range
                - end_date: End date for data range
                - account_id: Filter by account ID
                - meter_id: Filter by meter ID
                - limit: Maximum number of records to return
                
        Returns:
            DataFrame containing the fetched data
        """
        if not self.connected and not self.connect():
            return pd.DataFrame()
            
        try:
            params = params or {}
            
            if self.system_type in ['oracle_ccb', 'sql']:
                # For SQL-based systems, execute the query
                # Process parameters to replace placeholders
                formatted_query = query
                
                if '{start_date}' in formatted_query and 'start_date' in params:
                    formatted_query = formatted_query.replace('{start_date}', params['start_date'])
                if '{end_date}' in formatted_query and 'end_date' in params:
                    formatted_query = formatted_query.replace('{end_date}', params['end_date'])
                if '{account_id}' in formatted_query and 'account_id' in params:
                    formatted_query = formatted_query.replace('{account_id}', params['account_id'])
                if '{meter_id}' in formatted_query and 'meter_id' in params:
                    formatted_query = formatted_query.replace('{meter_id}', params['meter_id'])
                if '{limit}' in formatted_query and 'limit' in params:
                    formatted_query = formatted_query.replace('{limit}', str(params['limit']))
                
                # Execute the query and return as DataFrame
                return pd.read_sql(formatted_query, self.connection)
                
            elif self.system_type == 'sap_isu':
                # For SAP ISU, we would execute a BAPI or RFC call
                # This is a simplified placeholder implementation
                logger.warning("SAP ISU data fetch is not fully implemented")
                
                # Return an empty DataFrame with expected columns based on query type
                if 'invoice' in query.lower():
                    return pd.DataFrame(columns=[
                        'INVOICE_ID', 'ACCOUNT_ID', 'INVOICE_DATE', 'AMOUNT', 'STATUS'
                    ])
                elif 'account' in query.lower():
                    return pd.DataFrame(columns=[
                        'ACCOUNT_ID', 'CUSTOMER_NAME', 'ADDRESS', 'CATEGORY', 'STATUS'
                    ])
                elif 'meter' in query.lower():
                    return pd.DataFrame(columns=[
                        'METER_ID', 'ACCOUNT_ID', 'INSTALLATION_DATE', 'METER_TYPE', 'STATUS'
                    ])
                elif 'reading' in query.lower():
                    return pd.DataFrame(columns=[
                        'READING_ID', 'METER_ID', 'READING_DATE', 'READING_VALUE', 'READING_TYPE'
                    ])
                else:
                    return pd.DataFrame()
                    
            elif self.system_type in ['dynamics', 'custom_api']:
                # For API-based systems, call the appropriate endpoint
                url = f"{self.connection['base_url']}/{query}"
                
                # Add query parameters
                request_params = {}
                
                if 'start_date' in params:
                    request_params['startDate'] = params['start_date']
                if 'end_date' in params:
                    request_params['endDate'] = params['end_date']
                if 'account_id' in params:
                    request_params['accountId'] = params['account_id']
                if 'meter_id' in params:
                    request_params['meterId'] = params['meter_id']
                if 'limit' in params:
                    request_params['limit'] = params['limit']
                
                headers = self.connection.get('headers', {})
                
                response = requests.get(url, headers=headers, params=request_params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle different response structures
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    elif isinstance(data, dict):
                        if 'results' in data:
                            return pd.DataFrame(data['results'])
                        elif 'data' in data:
                            return pd.DataFrame(data['data'])
                        elif 'items' in data:
                            return pd.DataFrame(data['items'])
                        else:
                            # Single record response
                            return pd.DataFrame([data])
                    else:
                        logger.error(f"Unexpected API response format: {type(data)}")
                        return pd.DataFrame()
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return pd.DataFrame()
            
            # If we reach here, the system type is not supported
            logger.warning(f"Fetch not fully implemented for billing system: {self.system_type}")
            return pd.DataFrame()
            
        except Exception as e:
            self.handle_error(e, "fetch_data")
            return pd.DataFrame()
            
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of available datasets/tables from the billing system
        
        Returns:
            List of dictionaries with dataset metadata including:
            - id: Dataset/table identifier
            - name: Human-readable name
            - description: Description if available
            - type: Data type/category (e.g., 'account', 'invoice', 'meter', 'reading')
            - update_frequency: Update frequency if available
        """
        if not self.connected and not self.connect():
            return []
            
        try:
            if self.system_type == 'oracle_ccb':
                # Get tables from Oracle schema
                query = """
                SELECT 
                    TABLE_NAME as name,
                    'TABLE' as object_type
                FROM ALL_TABLES
                WHERE OWNER = :owner
                ORDER BY TABLE_NAME
                """
                
                cursor = self.connection.cursor()
                cursor.execute(query, owner=self.connection_params.get('username').upper())
                
                tables = []
                for row in cursor:
                    tables.append({
                        'id': row[0],
                        'name': row[0],
                        'description': '',
                        'type': 'table',
                        'update_frequency': 'unknown'
                    })
                
                cursor.close()
                return tables
                
            elif self.system_type == 'sql':
                # Get tables from SQL database
                if 'SQL Server' in self.connection_params.get('driver', 'SQL Server'):
                    # SQL Server
                    query = """
                    SELECT 
                        TABLE_SCHEMA + '.' + TABLE_NAME as full_name,
                        TABLE_NAME as name,
                        TABLE_SCHEMA as schema_name,
                        'TABLE' as object_type
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_SCHEMA, TABLE_NAME
                    """
                else:
                    # Generic SQL (MySQL, PostgreSQL, etc.)
                    query = """
                    SELECT 
                        TABLE_SCHEMA || '.' || TABLE_NAME as full_name,
                        TABLE_NAME as name,
                        TABLE_SCHEMA as schema_name,
                        'TABLE' as object_type
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_SCHEMA, TABLE_NAME
                    """
                
                df = pd.read_sql(query, self.connection)
                
                tables = []
                for _, row in df.iterrows():
                    tables.append({
                        'id': row['full_name'],
                        'name': row['name'],
                        'description': f"Schema: {row['schema_name']}",
                        'type': 'table',
                        'update_frequency': 'unknown'
                    })
                
                return tables
                
            elif self.system_type in ['dynamics', 'custom_api']:
                # For API-based systems, fetch available endpoints/entities
                url = f"{self.connection['base_url']}/metadata"
                
                headers = self.connection.get('headers', {})
                
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Process based on API structure
                        if 'entities' in data:
                            entities = data['entities']
                            return [
                                {
                                    'id': entity.get('path', entity['name']),
                                    'name': entity['name'],
                                    'description': entity.get('description', ''),
                                    'type': entity.get('type', 'entity'),
                                    'update_frequency': entity.get('updateFrequency', 'unknown')
                                }
                                for entity in entities
                            ]
                        elif 'endpoints' in data:
                            endpoints = data['endpoints']
                            return [
                                {
                                    'id': endpoint['path'],
                                    'name': endpoint.get('name', endpoint['path']),
                                    'description': endpoint.get('description', ''),
                                    'type': endpoint.get('type', 'endpoint'),
                                    'update_frequency': endpoint.get('updateFrequency', 'unknown')
                                }
                                for endpoint in endpoints
                            ]
                        else:
                            # Fallback to common endpoints for billing systems
                            return [
                                {
                                    'id': 'accounts',
                                    'name': 'Customer Accounts',
                                    'description': 'Customer account information',
                                    'type': 'account',
                                    'update_frequency': 'daily'
                                },
                                {
                                    'id': 'invoices',
                                    'name': 'Customer Invoices',
                                    'description': 'Billing invoices',
                                    'type': 'invoice',
                                    'update_frequency': 'daily'
                                },
                                {
                                    'id': 'meters',
                                    'name': 'Water Meters',
                                    'description': 'Meter installation and configuration',
                                    'type': 'meter',
                                    'update_frequency': 'daily'
                                },
                                {
                                    'id': 'readings',
                                    'name': 'Meter Readings',
                                    'description': 'Water consumption readings',
                                    'type': 'reading',
                                    'update_frequency': 'daily'
                                },
                                {
                                    'id': 'payments',
                                    'name': 'Customer Payments',
                                    'description': 'Payment transactions',
                                    'type': 'payment',
                                    'update_frequency': 'daily'
                                }
                            ]
                    else:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        # Fallback to common endpoints
                        return self._get_default_billing_datasets()
                except Exception as e:
                    logger.error(f"Error fetching API metadata: {str(e)}")
                    # Fallback to common endpoints
                    return self._get_default_billing_datasets()
                    
            elif self.system_type == 'sap_isu':
                # For SAP ISU, return common tables/entities
                return self._get_default_billing_datasets()
            
            # Return empty list for unsupported system types
            logger.warning(f"Getting available datasets not fully implemented for system: {self.system_type}")
            return []
            
        except Exception as e:
            self.handle_error(e, "get_available_datasets")
            return []
    
    def _get_default_billing_datasets(self) -> List[Dict[str, Any]]:
        """
        Return a default list of common billing system datasets
        
        Returns:
            List of dictionary objects with dataset metadata
        """
        return [
            {
                'id': 'accounts',
                'name': 'Customer Accounts',
                'description': 'Customer account information',
                'type': 'account',
                'update_frequency': 'daily'
            },
            {
                'id': 'invoices',
                'name': 'Customer Invoices',
                'description': 'Billing invoices',
                'type': 'invoice',
                'update_frequency': 'daily'
            },
            {
                'id': 'meters',
                'name': 'Water Meters',
                'description': 'Meter installation and configuration',
                'type': 'meter',
                'update_frequency': 'daily'
            },
            {
                'id': 'readings',
                'name': 'Meter Readings',
                'description': 'Water consumption readings',
                'type': 'reading',
                'update_frequency': 'daily'
            },
            {
                'id': 'payments',
                'name': 'Customer Payments',
                'description': 'Payment transactions',
                'type': 'payment',
                'update_frequency': 'daily'
            },
            {
                'id': 'rates',
                'name': 'Rate Plans',
                'description': 'Water usage rate plans',
                'type': 'rate',
                'update_frequency': 'monthly'
            }
        ]
    
    def get_customer_data(self, account_id: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Fetch customer account data
        
        Args:
            account_id: Optional specific account ID
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with customer data
        """
        if not self.connected and not self.connect():
            return pd.DataFrame()
            
        try:
            if self.system_type in ['oracle_ccb', 'sql']:
                # SQL query for customer data
                if account_id:
                    query = f"""
                    SELECT * FROM customer_accounts
                    WHERE account_id = '{account_id}'
                    """
                else:
                    query = f"""
                    SELECT * FROM customer_accounts
                    ORDER BY account_id
                    LIMIT {limit}
                    """
                    
                return self.fetch_data(query)
                
            elif self.system_type in ['dynamics', 'custom_api']:
                # API endpoint for customer data
                endpoint = "accounts"
                params = {'limit': limit}
                
                if account_id:
                    endpoint = f"accounts/{account_id}"
                    
                return self.fetch_data(endpoint, params)
                
            else:
                logger.warning(f"Get customer data not fully implemented for system: {self.system_type}")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e, "get_customer_data")
            return pd.DataFrame()
    
    def get_consumption_data(self, meter_id: Optional[str] = None, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Fetch water consumption data
        
        Args:
            meter_id: Optional specific meter ID
            start_date: Start date for data range (YYYY-MM-DD)
            end_date: End date for data range (YYYY-MM-DD)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with consumption data
        """
        if not self.connected and not self.connect():
            return pd.DataFrame()
            
        try:
            params = {
                'limit': limit
            }
            
            if meter_id:
                params['meter_id'] = meter_id
                
            if start_date:
                params['start_date'] = start_date
                
            if end_date:
                params['end_date'] = end_date
                
            if self.system_type in ['oracle_ccb', 'sql']:
                # SQL query for consumption data
                query = """
                SELECT 
                    reading_id,
                    meter_id,
                    reading_date,
                    reading_value,
                    previous_reading,
                    consumption,
                    reading_type,
                    reader_id
                FROM meter_readings
                WHERE 1=1
                """
                
                if meter_id:
                    query += f" AND meter_id = '{meter_id}'"
                    
                if start_date:
                    query += f" AND reading_date >= '{start_date}'"
                    
                if end_date:
                    query += f" AND reading_date <= '{end_date}'"
                    
                query += f" ORDER BY meter_id, reading_date DESC LIMIT {limit}"
                
                return self.fetch_data(query)
                
            elif self.system_type in ['dynamics', 'custom_api']:
                # API endpoint for consumption data
                endpoint = "readings"
                
                if meter_id:
                    endpoint = f"meters/{meter_id}/readings"
                    
                return self.fetch_data(endpoint, params)
                
            else:
                logger.warning(f"Get consumption data not fully implemented for system: {self.system_type}")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e, "get_consumption_data")
            return pd.DataFrame()
    
    def get_billing_data(self, account_id: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: int = 1000) -> pd.DataFrame:
        """
        Fetch billing invoice data
        
        Args:
            account_id: Optional specific account ID
            start_date: Start date for data range (YYYY-MM-DD)
            end_date: End date for data range (YYYY-MM-DD)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with billing data
        """
        if not self.connected and not self.connect():
            return pd.DataFrame()
            
        try:
            params = {
                'limit': limit
            }
            
            if account_id:
                params['account_id'] = account_id
                
            if start_date:
                params['start_date'] = start_date
                
            if end_date:
                params['end_date'] = end_date
                
            if self.system_type in ['oracle_ccb', 'sql']:
                # SQL query for billing data
                query = """
                SELECT 
                    invoice_id,
                    account_id,
                    customer_name,
                    invoice_date,
                    due_date,
                    total_amount,
                    consumption,
                    status,
                    payment_date
                FROM invoices
                WHERE 1=1
                """
                
                if account_id:
                    query += f" AND account_id = '{account_id}'"
                    
                if start_date:
                    query += f" AND invoice_date >= '{start_date}'"
                    
                if end_date:
                    query += f" AND invoice_date <= '{end_date}'"
                    
                query += f" ORDER BY account_id, invoice_date DESC LIMIT {limit}"
                
                return self.fetch_data(query)
                
            elif self.system_type in ['dynamics', 'custom_api']:
                # API endpoint for billing data
                endpoint = "invoices"
                
                if account_id:
                    endpoint = f"accounts/{account_id}/invoices"
                    
                return self.fetch_data(endpoint, params)
                
            else:
                logger.warning(f"Get billing data not fully implemented for system: {self.system_type}")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e, "get_billing_data")
            return pd.DataFrame()
    
    def get_payment_data(self, account_id: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """
        Fetch payment transaction data
        
        Args:
            account_id: Optional specific account ID
            start_date: Start date for data range (YYYY-MM-DD)
            end_date: End date for data range (YYYY-MM-DD)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with payment data
        """
        if not self.connected and not self.connect():
            return pd.DataFrame()
            
        try:
            params = {
                'limit': limit
            }
            
            if account_id:
                params['account_id'] = account_id
                
            if start_date:
                params['start_date'] = start_date
                
            if end_date:
                params['end_date'] = end_date
                
            if self.system_type in ['oracle_ccb', 'sql']:
                # SQL query for payment data
                query = """
                SELECT 
                    payment_id,
                    account_id,
                    invoice_id,
                    payment_date,
                    amount,
                    payment_method,
                    status,
                    reference_number
                FROM payments
                WHERE 1=1
                """
                
                if account_id:
                    query += f" AND account_id = '{account_id}'"
                    
                if start_date:
                    query += f" AND payment_date >= '{start_date}'"
                    
                if end_date:
                    query += f" AND payment_date <= '{end_date}'"
                    
                query += f" ORDER BY account_id, payment_date DESC LIMIT {limit}"
                
                return self.fetch_data(query)
                
            elif self.system_type in ['dynamics', 'custom_api']:
                # API endpoint for payment data
                endpoint = "payments"
                
                if account_id:
                    endpoint = f"accounts/{account_id}/payments"
                    
                return self.fetch_data(endpoint, params)
                
            else:
                logger.warning(f"Get payment data not fully implemented for system: {self.system_type}")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e, "get_payment_data")
            return pd.DataFrame()
