# Base connector for external data systems

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseConnector(ABC):
    """Abstract base class for all data connectors"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize connector with connection parameters
        
        Args:
            connection_params: Dictionary of connection parameters such as:
                - host: Host address
                - port: Port number
                - username: Authentication username
                - password: Authentication password
                - api_key: API key if applicable
                - cert_path: Path to certificate file if needed
                - other connector-specific parameters
        """
        self.connection_params = connection_params
        self.connection = None
        self.connected = False
        self.last_error = None
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to the data source
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the connection is working
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        pass
    
    @abstractmethod
    def fetch_data(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from the source using the provided query
        
        Args:
            query: Query string or identifier for the data to fetch
            params: Optional parameters for the query
            
        Returns:
            DataFrame containing the fetched data
        """
        pass
    
    @abstractmethod
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of available datasets from the source
        
        Returns:
            List of dictionaries with dataset metadata
        """
        pass
    
    def log_activity(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log connector activity
        
        Args:
            action: Description of the action being performed
            details: Additional details about the action
        """
        log_entry = {
            "connector_type": self.__class__.__name__,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        logger.info(f"Connector activity: {json.dumps(log_entry)}")
    
    def handle_error(self, error: Exception, context: str) -> None:
        """
        Handle and log connector errors
        
        Args:
            error: The exception that occurred
            context: Description of what was happening when the error occurred
        """
        self.last_error = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.error(f"Connector error in {context}: {str(error)}")
