# Data validators for water utility data

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import logging
import re
from datetime import datetime, timedelta
import great_expectations as ge
from great_expectations.dataset import PandasDataset
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, DataQualityProfileSection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Base class for all data validators"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with configuration
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.validation_results = {}
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the provided data
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        raise NotImplementedError("Subclasses must implement validate()")
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results
        
        Returns:
            Dictionary with validation summary
        """
        passed = sum(1 for result in self.validation_results.values() if result.get('status') == 'passed')
        total = len(self.validation_results)
        
        return {
            'total_checks': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': (passed / total) if total > 0 else 0,
            'checks': self.validation_results
        }


class SchemaValidator(DataValidator):
    """Validator for checking data schema"""
    
    def __init__(self, schema: Dict[str, Any], **kwargs):
        """
        Initialize schema validator
        
        Args:
            schema: Dictionary defining expected schema:
                - columns: List of expected columns
                - types: Dictionary mapping columns to expected types
                - required: List of required columns
                - nullable: List of columns that can have null values
        """
        super().__init__(**kwargs)
        self.schema = schema
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against expected schema
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {}
        
        # Check required columns
        missing_columns = []
        if 'required' in self.schema:
            for col in self.schema['required']:
                if col not in data.columns:
                    missing_columns.append(col)
                    
        self.validation_results['required_columns'] = {
            'status': 'passed' if not missing_columns else 'failed',
            'message': 'All required columns present' if not missing_columns else f"Missing required columns: {missing_columns}",
            'missing_columns': missing_columns
        }
        
        # Check column types
        type_errors = []
        if 'types' in self.schema:
            for col, expected_type in self.schema['types'].items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    # Check if types are compatible
                    if not self._check_types_compatible(actual_type, expected_type):
                        type_errors.append({
                            'column': col,
                            'expected_type': expected_type,
                            'actual_type': actual_type
                        })
                        
        self.validation_results['column_types'] = {
            'status': 'passed' if not type_errors else 'failed',
            'message': 'All column types match expected types' if not type_errors else f"Column type mismatches found: {len(type_errors)}",
            'type_errors': type_errors
        }
        
        # Check for null values in non-nullable columns
        null_errors = []
        if 'nullable' in self.schema:
            nullable_cols = self.schema['nullable']
            for col in data.columns:
                if col not in nullable_cols and col in data.columns and data[col].isnull().any():
                    null_count = data[col].isnull().sum()
                    null_errors.append({
                        'column': col,
                        'null_count': int(null_count),
                        'null_percent': float(null_count / len(data) * 100)
                    })
                    
        self.validation_results['null_checks'] = {
            'status': 'passed' if not null_errors else 'failed',
            'message': 'No unexpected null values found' if not null_errors else f"Unexpected null values found in {len(null_errors)} columns",
            'null_errors': null_errors
        }
        
        # Check for extra columns
        extra_columns = []
        if 'columns' in self.schema:
            expected_columns = set(self.schema['columns'])
            actual_columns = set(data.columns)
            extra_columns = list(actual_columns - expected_columns)
            
        self.validation_results['extra_columns'] = {
            'status': 'passed' if not extra_columns else 'warning',
            'message': 'No extra columns found' if not extra_columns else f"Extra columns found: {extra_columns}",
            'extra_columns': extra_columns
        }
        
        return self.get_validation_summary()
    
    def _check_types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """
        Check if actual data type is compatible with expected type
        
        Args:
            actual_type: Actual data type
            expected_type: Expected data type
            
        Returns:
            True if types are compatible, False otherwise
        """
        # Convert pandas/numpy types to more general categories
        type_mappings = {
            'int64': 'integer',
            'int32': 'integer',
            'int16': 'integer',
            'int8': 'integer',
            'float64': 'float',
            'float32': 'float',
            'object': 'string',
            'bool': 'boolean',
            'datetime64[ns]': 'datetime',
            'category': 'string'
        }
        
        actual_general = type_mappings.get(actual_type, actual_type)
        expected_general = type_mappings.get(expected_type, expected_type)
        
        # Check for direct match
        if actual_general == expected_general:
            return True
            
        # Check for numeric compatibility
        if actual_general in ['integer', 'float'] and expected_general in ['integer', 'float']:
            return True
            
        return False


class ValueRangeValidator(DataValidator):
    """Validator for checking value ranges"""
    
    def __init__(self, ranges: Dict[str, Dict[str, Any]], **kwargs):
        """
        Initialize value range validator
        
        Args:
            ranges: Dictionary mapping columns to range constraints:
                - min: Minimum allowed value
                - max: Maximum allowed value
                - allowed_values: List of allowed values (for categorical)
        """
        super().__init__(**kwargs)
        self.ranges = ranges
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against value range constraints
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {}
        
        for col, constraints in self.ranges.items():
            if col not in data.columns:
                self.validation_results[f"range_{col}"] = {
                    'status': 'skipped',
                    'message': f"Column '{col}' not found in data"
                }
                continue
                
            # Check minimum value
            min_errors = 0
            if 'min' in constraints:
                min_val = constraints['min']
                min_errors = (data[col] < min_val).sum()
                self.validation_results[f"min_{col}"] = {
                    'status': 'passed' if min_errors == 0 else 'failed',
                    'message': f"All values in '{col}' meet minimum ({min_val})" if min_errors == 0 else f"{min_errors} values in '{col}' below minimum ({min_val})",
                    'error_count': int(min_errors),
                    'error_percent': float(min_errors / len(data) * 100) if len(data) > 0 else 0
                }
                
            # Check maximum value
            max_errors = 0
            if 'max' in constraints:
                max_val = constraints['max']
                max_errors = (data[col] > max_val).sum()
                self.validation_results[f"max_{col}"] = {
                    'status': 'passed' if max_errors == 0 else 'failed',
                    'message': f"All values in '{col}' meet maximum ({max_val})" if max_errors == 0 else f"{max_errors} values in '{col}' above maximum ({max_val})",
                    'error_count': int(max_errors),
                    'error_percent': float(max_errors / len(data) * 100) if len(data) > 0 else 0
                }
                
            # Check allowed values
            if 'allowed_values' in constraints:
                allowed = set(constraints['allowed_values'])
                # For non-numeric columns, convert to string for comparison
                if data[col].dtype == 'object' or data[col].dtype == 'category':
                    invalid_values = data[col].astype(str).apply(lambda x: x not in allowed)
                else:
                    invalid_values = data[col].apply(lambda x: x not in allowed)
                    
                invalid_count = invalid_values.sum()
                self.validation_results[f"allowed_{col}"] = {
                    'status': 'passed' if invalid_count == 0 else 'failed',
                    'message': f"All values in '{col}' are in the allowed set" if invalid_count == 0 else f"{invalid_count} values in '{col}' not in allowed set",
                    'error_count': int(invalid_count),
                    'error_percent': float(invalid_count / len(data) * 100) if len(data) > 0 else 0,
                    'invalid_examples': list(data.loc[invalid_values, col].unique())[:5]  # Sample of invalid values
                }
                
        return self.get_validation_summary()


class FormatValidator(DataValidator):
    """Validator for checking data formats (dates, email, etc.)"""
    
    def __init__(self, format_specs: Dict[str, Dict[str, Any]], **kwargs):
        """
        Initialize format validator
        
        Args:
            format_specs: Dictionary mapping columns to format specifications:
                - type: Format type ('date', 'email', 'phone', 'url', 'regex')
                - pattern: Regex pattern for 'regex' type
                - date_format: Expected date format for 'date' type
        """
        super().__init__(**kwargs)
        self.format_specs = format_specs
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against format specifications
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {}
        
        for col, spec in self.format_specs.items():
            if col not in data.columns:
                self.validation_results[f"format_{col}"] = {
                    'status': 'skipped',
                    'message': f"Column '{col}' not found in data"
                }
                continue
                
            format_type = spec.get('type', 'regex')
            
            # Skip null values in validation
            non_null_data = data[col].dropna()
            if len(non_null_data) == 0:
                self.validation_results[f"format_{col}"] = {
                    'status': 'skipped',
                    'message': f"Column '{col}' contains only null values"
                }
                continue
                
            # Convert numeric columns to string for regex validation
            if format_type in ['email', 'phone', 'url', 'regex']:
                values_to_check = non_null_data.astype(str)
            else:
                values_to_check = non_null_data
                
            if format_type == 'date':
                # Date format validation
                date_format = spec.get('date_format', '%Y-%m-%d')
                
                if values_to_check.dtype == 'datetime64[ns]':
                    # Already a datetime column
                    invalid_count = 0
                else:
                    # Try to parse as datetime
                    try:
                        invalid_mask = values_to_check.apply(lambda x: not self._is_valid_date(x, date_format))
                        invalid_count = invalid_mask.sum()
                    except:
                        invalid_count = len(values_to_check)  # If exception, all are invalid
                        
            elif format_type == 'email':
                # Email validation
                pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_mask = ~values_to_check.str.match(pattern)
                invalid_count = invalid_mask.sum()
                
            elif format_type == 'phone':
                # Phone number validation (simple pattern)
                pattern = r'^\+?[0-9]{10,15}$'
                invalid_mask = ~values_to_check.str.match(pattern)
                invalid_count = invalid_mask.sum()
                
            elif format_type == 'url':
                # URL validation
                pattern = r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
                invalid_mask = ~values_to_check.str.match(pattern)
                invalid_count = invalid_mask.sum()
                
            elif format_type == 'regex':
                # Custom regex validation
                pattern = spec.get('pattern', '')
                if not pattern:
                    self.validation_results[f"format_{col}"] = {
                        'status': 'skipped',
                        'message': f"No pattern provided for regex validation of '{col}'"
                    }
                    continue
                    
                try:
                    invalid_mask = ~values_to_check.str.match(pattern)
                    invalid_count = invalid_mask.sum()
                except Exception as e:
                    self.validation_results[f"format_{col}"] = {
                        'status': 'error',
                        'message': f"Error validating '{col}' with regex: {str(e)}"
                    }
                    continue
            else:
                self.validation_results[f"format_{col}"] = {
                    'status': 'skipped',
                    'message': f"Unknown format type '{format_type}' for '{col}'"
                }
                continue
                
            # Store validation result
            invalid_percent = 100 * invalid_count / len(values_to_check) if len(values_to_check) > 0 else 0
            self.validation_results[f"format_{col}"] = {
                'status': 'passed' if invalid_count == 0 else 'failed',
                'message': f"All values in '{col}' match required format" if invalid_count == 0 else f"{invalid_count} values ({invalid_percent:.1f}%) in '{col}' don't match required format",
                'error_count': int(invalid_count),
                'error_percent': float(invalid_percent)
            }
            
        return self.get_validation_summary()
    
    def _is_valid_date(self, date_str: str, date_format: str) -> bool:
        """
        Check if string is a valid date in the specified format
        
        Args:
            date_str: Date string to validate
            date_format: Expected date format
            
        Returns:
            True if valid date, False otherwise
        """
        try:
            datetime.strptime(str(date_str), date_format)
            return True
        except ValueError:
            return False


class RelationshipValidator(DataValidator):
    """Validator for checking relationships between datasets"""
    
    def __init__(self, relationships: List[Dict[str, Any]], **kwargs):
        """
        Initialize relationship validator
        
        Args:
            relationships: List of relationship definitions:
                - source_table: Name of source table
                - source_col: Column in source table
                - target_table: Name of target table
                - target_col: Column in target table
                - type: Relationship type ('one-to-one', 'one-to-many', 'many-to-one')
        """
        super().__init__(**kwargs)
        self.relationships = relationships
        
    def validate(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate relationships between datasets
        
        Args:
            data_dict: Dictionary mapping table names to DataFrames
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {}
        
        for i, rel in enumerate(self.relationships):
            source_table = rel.get('source_table')
            target_table = rel.get('target_table')
            source_col = rel.get('source_col')
            target_col = rel.get('target_col')
            rel_type = rel.get('type', 'one-to-many')
            
            # Check if tables and columns exist
            if source_table not in data_dict:
                self.validation_results[f"rel_{i}"] = {
                    'status': 'skipped',
                    'message': f"Source table '{source_table}' not found in data"
                }
                continue
                
            if target_table not in data_dict:
                self.validation_results[f"rel_{i}"] = {
                    'status': 'skipped',
                    'message': f"Target table '{target_table}' not found in data"
                }
                continue
                
            if source_col not in data_dict[source_table].columns:
                self.validation_results[f"rel_{i}"] = {
                    'status': 'skipped',
                    'message': f"Source column '{source_col}' not found in '{source_table}'"
                }
                continue
                
            if target_col not in data_dict[target_table].columns:
                self.validation_results[f"rel_{i}"] = {
                    'status': 'skipped',
                    'message': f"Target column '{target_col}' not found in '{target_table}'"
                }
                continue
                
            # Get unique values from each side
            source_values = set(data_dict[source_table][source_col].dropna().unique())
            target_values = set(data_dict[target_table][target_col].dropna().unique())
            
            # Find orphaned values (in source but not in target)
            orphaned = source_values - target_values
            orphaned_count = len(orphaned)
            
            # Find invalid references (in target but not in source) for many-to-one
            invalid_refs = target_values - source_values if rel_type == 'many-to-one' else set()
            invalid_count = len(invalid_refs)
            
            # Check cardinality for one-to-one
            cardinality_errors = 0
            if rel_type == 'one-to-one':
                # In one-to-one, both source and target values should be unique
                source_dupes = data_dict[source_table][source_col].value_counts()
                source_dupes = source_dupes[source_dupes > 1]
                
                target_dupes = data_dict[target_table][target_col].value_counts()
                target_dupes = target_dupes[target_dupes > 1]
                
                cardinality_errors = len(source_dupes) + len(target_dupes)
                
            # Store validation result
            self.validation_results[f"rel_{i}"] = {
                'relationship': f"{source_table}.{source_col} -> {target_table}.{target_col} ({rel_type})",
                'status': 'passed' if orphaned_count == 0 and invalid_count == 0 and cardinality_errors == 0 else 'failed',
                'orphaned_count': orphaned_count,
                'invalid_reference_count': invalid_count,
                'cardinality_error_count': cardinality_errors,
                'message': self._format_rel_message(orphaned_count, invalid_count, cardinality_errors, rel_type)
            }
            
        return self.get_validation_summary()
    
    def _format_rel_message(self, orphaned: int, invalid: int, cardinality: int, rel_type: str) -> str:
        """
        Format validation message for relationship
        
        Args:
            orphaned: Number of orphaned records
            invalid: Number of invalid references
            cardinality: Number of cardinality errors
            rel_type: Relationship type
            
        Returns:
            Formatted message
        """
        if orphaned == 0 and invalid == 0 and cardinality == 0:
            return "Relationship is valid"
            
        errors = []
        if orphaned > 0:
            errors.append(f"{orphaned} orphaned records")
            
        if invalid > 0:
            errors.append(f"{invalid} invalid references")
            
        if cardinality > 0 and rel_type == 'one-to-one':
            errors.append(f"{cardinality} cardinality violations")
            
        return "Relationship errors: " + ", ".join(errors)


class ComplexValidator(DataValidator):
    """Validator for complex business rules and cross-field validations"""
    
    def __init__(self, rules: List[Dict[str, Any]], **kwargs):
        """
        Initialize complex validator
        
        Args:
            rules: List of business rule definitions:
                - name: Rule name
                - description: Rule description
                - condition: String representation of condition (Python expression)
                - columns: List of columns used in the condition
        """
        super().__init__(**kwargs)
        self.rules = rules
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against complex business rules
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {}
        
        for i, rule in enumerate(self.rules):
            rule_name = rule.get('name', f"rule_{i}")
            description = rule.get('description', 'No description')
            condition = rule.get('condition', '')
            columns = rule.get('columns', [])
            
            # Check if required columns exist
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                self.validation_results[rule_name] = {
                    'status': 'skipped',
                    'message': f"Missing required columns for rule: {missing_cols}"
                }
                continue
                
            if not condition:
                self.validation_results[rule_name] = {
                    'status': 'skipped',
                    'message': "No condition provided for rule"
                }
                continue
                
            try:
                # Evaluate the condition
                # Note: This uses eval which can be dangerous if used with untrusted input
                # In a production system, use a safer alternative like numexpr
                result = data.eval(condition)
                violation_count = (~result).sum()
                
                self.validation_results[rule_name] = {
                    'status': 'passed' if violation_count == 0 else 'failed',
                    'message': f"All records pass rule '{description}'" if violation_count == 0 else f"{violation_count} records violate rule '{description}'",
                    'violation_count': int(violation_count),
                    'violation_percent': float(violation_count / len(data) * 100) if len(data) > 0 else 0
                }
                
            except Exception as e:
                self.validation_results[rule_name] = {
                    'status': 'error',
                    'message': f"Error evaluating rule: {str(e)}"
                }
                
        return self.get_validation_summary()


class TimeSeriesValidator(DataValidator):
    """Validator for time series data"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize time series validator
        
        Args:
            config: Configuration for time series validation:
                - time_col: Name of timestamp column
                - value_col: Name of value column
                - frequency: Expected frequency ('daily', 'hourly', etc.)
                - max_gap: Maximum allowed gap in time series
                - detect_anomalies: Whether to detect anomalies
                - anomaly_threshold: Z-score threshold for anomalies
        """
        super().__init__(config, **kwargs)
        self.time_col = config.get('time_col', 'timestamp')
        self.value_col = config.get('value_col', 'value')
        self.frequency = config.get('frequency', 'daily')
        self.max_gap = config.get('max_gap', 1)
        self.detect_anomalies = config.get('detect_anomalies', True)
        self.anomaly_threshold = config.get('anomaly_threshold', 3.0)
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate time series data
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {}
        
        # Check if required columns exist
        if self.time_col not in data.columns:
            self.validation_results['time_column'] = {
                'status': 'skipped',
                'message': f"Time column '{self.time_col}' not found in data"
            }
            return self.get_validation_summary()
            
        if self.value_col not in data.columns:
            self.validation_results['value_column'] = {
                'status': 'skipped',
                'message': f"Value column '{self.value_col}' not found in data"
            }
            return self.get_validation_summary()
            
        # Convert time column to datetime if needed
        if data[self.time_col].dtype != 'datetime64[ns]':
            try:
                data = data.copy()
                data[self.time_col] = pd.to_datetime(data[self.time_col])
            except Exception as e:
                self.validation_results['time_format'] = {
                    'status': 'error',
                    'message': f"Error converting time column to datetime: {str(e)}"
                }
                return self.get_validation_summary()
                
        # Sort by time column
        data = data.sort_values(by=self.time_col)
        
        # Check for duplicates
        duplicate_times = data[self.time_col].duplicated()
        duplicate_count = duplicate_times.sum()
        self.validation_results['unique_timestamps'] = {
            'status': 'passed' if duplicate_count == 0 else 'failed',
            'message': 'All timestamps are unique' if duplicate_count == 0 else f"{duplicate_count} duplicate timestamps found",
            'duplicate_count': int(duplicate_count)
        }
        
        # Check for gaps
        if len(data) >= 2:
            freq_map = {
                'hourly': pd.Timedelta(hours=1),
                'daily': pd.Timedelta(days=1),
                'weekly': pd.Timedelta(weeks=1),
                'monthly': pd.Timedelta(days=30),  # Approximate
                'yearly': pd.Timedelta(days=365)  # Approximate
            }
            
            expected_gap = freq_map.get(self.frequency, pd.Timedelta(days=1))
            max_allowed_gap = expected_gap * self.max_gap
            
            # Calculate gaps
            data = data.drop_duplicates(subset=[self.time_col]).copy()
            data['next_time'] = data[self.time_col].shift(-1)
            data['gap'] = data['next_time'] - data[self.time_col]
            
            # Find large gaps
            large_gaps = data[(data['gap'] > max_allowed_gap) & data['next_time'].notna()]
            large_gap_count = len(large_gaps)
            
            self.validation_results['time_series_continuity'] = {
                'status': 'passed' if large_gap_count == 0 else 'failed',
                'message': 'No large gaps in time series' if large_gap_count == 0 else f"{large_gap_count} large gaps found in time series",
                'gap_count': large_gap_count,
                'max_gap': str(large_gaps['gap'].max()) if large_gap_count > 0 else '0 days'
            }
            
        # Check for anomalies
        if self.detect_anomalies and len(data) >= 10:
            try:
                # Calculate z-scores
                data = data.drop_duplicates(subset=[self.time_col]).copy()
                data['value'] = pd.to_numeric(data[self.value_col], errors='coerce')
                data['z_score'] = (data['value'] - data['value'].mean()) / data['value'].std()
                
                # Identify anomalies
                anomalies = data[abs(data['z_score']) > self.anomaly_threshold]
                anomaly_count = len(anomalies)
                
                self.validation_results['anomalies'] = {
                    'status': 'passed' if anomaly_count == 0 else 'warning',
                    'message': 'No anomalies detected' if anomaly_count == 0 else f"{anomaly_count} potential anomalies detected",
                    'anomaly_count': anomaly_count,
                    'anomaly_percent': float(anomaly_count / len(data) * 100)
                }
            except Exception as e:
                self.validation_results['anomalies'] = {
                    'status': 'error',
                    'message': f"Error detecting anomalies: {str(e)}"
                }
                
        return self.get_validation_summary()


class DataQualityRunner:
    """Run multiple validators against a dataset"""
    
    def __init__(self):
        """Initialize data quality runner"""
        self.validators = []
        self.results = {}
        
    def add_validator(self, validator: DataValidator, name: Optional[str] = None) -> None:
        """
        Add a validator to the runner
        
        Args:
            validator: Validator instance
            name: Optional name for the validator
        """
        if name is None:
            name = f"validator_{len(self.validators)}"
            
        self.validators.append((name, validator))
        
    def run(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
           dataset_name: Optional[str] = "default") -> Dict[str, Any]:
        """
        Run all validators against the dataset
        
        Args:
            data: DataFrame or dictionary of DataFrames to validate
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary with validation results
        """
        start_time = datetime.now()
        
        if isinstance(data, pd.DataFrame):
            # Single DataFrame
            data_source = {'default': data}
        else:
            # Dictionary of DataFrames
            data_source = data
            
        all_results = {}
        
        for name, validator in self.validators:
            try:
                if isinstance(validator, RelationshipValidator):
                    # Relationship validator needs dictionary of DataFrames
                    result = validator.validate(data_source)
                else:
                    # Other validators work on a single DataFrame
                    result = validator.validate(data_source.get(dataset_name, pd.DataFrame()))
                    
                all_results[name] = result
            except Exception as e:
                all_results[name] = {
                    'status': 'error',
                    'message': f"Error running validator {name}: {str(e)}",
                    'error': str(e)
                }
                
        # Calculate overall status
        passed = sum(1 for result in all_results.values() 
                    if isinstance(result, dict) and result.get('total_checks', 0) > 0 
                    and result.get('passed', 0) == result.get('total_checks', 0))
        
        total = len(all_results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.results = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'validator_count': total,
            'passed_validators': passed,
            'overall_status': 'passed' if passed == total else 'failed',
            'validators': all_results
        }
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results
        
        Returns:
            Dictionary with validation summary
        """
        if not self.results:
            return {'status': 'not_run', 'message': 'Validation has not been run'}
            
        total_checks = 0
        passed_checks = 0
        
        for validator_name, result in self.results.get('validators', {}).items():
            if isinstance(result, dict):
                total_checks += result.get('total_checks', 0)
                passed_checks += result.get('passed', 0)
                
        return {
            'dataset': self.results.get('dataset'),
            'timestamp': self.results.get('timestamp'),
            'duration_seconds': self.results.get('duration_seconds'),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_rate': (passed_checks / total_checks) if total_checks > 0 else 0,
            'overall_status': 'passed' if passed_checks == total_checks else 'failed'
        }


# Pre-built validators for common water utility data

def get_meter_reading_validator() -> DataQualityRunner:
    """
    Get validator for water meter readings
    
    Returns:
        Configured DataQualityRunner
    """
    runner = DataQualityRunner()
    
    # Schema validation
    schema = {
        'columns': ['meter_id', 'reading_date', 'reading_value', 'reading_type', 'unit'],
        'required': ['meter_id', 'reading_date', 'reading_value'],
        'types': {
            'meter_id': 'string',
            'reading_date': 'datetime64[ns]',
            'reading_value': 'float',
            'reading_type': 'string',
            'unit': 'string'
        },
        'nullable': ['reading_type', 'unit']
    }
    runner.add_validator(SchemaValidator(schema), "schema")
    
    # Value range validation
    ranges = {
        'reading_value': {
            'min': 0,
            'max': 1000000  # Adjust based on expected max reading
        },
        'reading_type': {
            'allowed_values': ['manual', 'automatic', 'customer', 'estimated']
        },
        'unit': {
            'allowed_values': ['gallons', 'liters', 'cubic_meters', 'cubic_feet']
        }
    }
    runner.add_validator(ValueRangeValidator(ranges), "ranges")
    
    # Time series validation
    ts_config = {
        'time_col': 'reading_date',
        'value_col': 'reading_value',
        'frequency': 'monthly',  # Typical meter reading frequency
        'max_gap': 2,  # Allow up to 2x expected gap
        'detect_anomalies': True,
        'anomaly_threshold': 4.0  # Higher threshold for consumption which can vary widely
    }
    runner.add_validator(TimeSeriesValidator(ts_config), "time_series")
    
    # Format validation
    formats = {
        'meter_id': {
            'type': 'regex',
            'pattern': r'^[A-Z0-9]{5,20}$'  # Example pattern for meter IDs
        },
        'reading_date': {
            'type': 'date'
        }
    }
    runner.add_validator(FormatValidator(formats), "formats")
    
    return runner


def get_water_quality_validator() -> DataQualityRunner:
    """
    Get validator for water quality samples
    
    Returns:
        Configured DataQualityRunner
    """
    runner = DataQualityRunner()
    
    # Schema validation
    schema = {
        'columns': ['sample_id', 'location_id', 'sample_date', 'parameter', 'value', 'unit', 'method'],
        'required': ['sample_id', 'location_id', 'sample_date', 'parameter', 'value'],
        'types': {
            'sample_id': 'string',
            'location_id': 'string',
            'sample_date': 'datetime64[ns]',
            'parameter': 'string',
            'value': 'float',
            'unit': 'string',
            'method': 'string'
        },
        'nullable': ['unit', 'method']
    }
    runner.add_validator(SchemaValidator(schema), "schema")
    
    # Value range validation - different ranges for different parameters
    # This is simplified - would need to be expanded for all parameters
    ranges = {
        'value': {
            'min': 0  # Generic minimum for most parameters
        },
        'parameter': {
            'allowed_values': [
                'pH', 'turbidity', 'chlorine', 'lead', 'copper', 'bacteria', 
                'nitrate', 'nitrite', 'fluoride', 'hardness', 'iron', 'manganese'
            ]
        }
    }
    runner.add_validator(ValueRangeValidator(ranges), "ranges")
    
    # Complex validation for parameter-specific ranges
    rules = [
        {
            'name': 'ph_range',
            'description': 'pH must be between 6.5 and 8.5',
            'condition': "(parameter != 'pH') | ((value >= 6.5) & (value <= 8.5))",
            'columns': ['parameter', 'value']
        },
        {
            'name': 'turbidity_range',
            'description': 'Turbidity must be less than 5 NTU',
            'condition': "(parameter != 'turbidity') | (value < 5)",
            'columns': ['parameter', 'value']
        },
        {
            'name': 'chlorine_range',
            'description': 'Chlorine must be between 0.2 and 4.0 mg/L',
            'condition': "(parameter != 'chlorine') | ((value >= 0.2) & (value <= 4.0))",
            'columns': ['parameter', 'value']
        }
    ]
    runner.add_validator(ComplexValidator(rules), "parameter_rules")
    
    # Format validation
    formats = {
        'sample_id': {
            'type': 'regex',
            'pattern': r'^S[0-9]{8}$'  # Example pattern: S followed by 8 digits
        },
        'location_id': {
            'type': 'regex',
            'pattern': r'^L[0-9]{4}$'  # Example pattern: L followed by 4 digits
        },
        'sample_date': {
            'type': 'date'
        }
    }
    runner.add_validator(FormatValidator(formats), "formats")
    
    return runner


def get_customer_data_validator() -> DataQualityRunner:
    """
    Get validator for customer account data
    
    Returns:
        Configured DataQualityRunner
    """
    runner = DataQualityRunner()
    
    # Schema validation
    schema = {
        'columns': ['account_id', 'customer_name', 'address', 'city', 'state', 'postal_code', 
                   'email', 'phone', 'account_type', 'status', 'created_date'],
        'required': ['account_id', 'customer_name', 'address', 'city', 'state', 'postal_code', 'status'],
        'types': {
            'account_id': 'string',
            'customer_name': 'string',
            'address': 'string',
            'city': 'string',
            'state': 'string',
            'postal_code': 'string',
            'email': 'string',
            'phone': 'string',
            'account_type': 'string',
            'status': 'string',
            'created_date': 'datetime64[ns]'
        },
        'nullable': ['email', 'phone', 'account_type']
    }
    runner.add_validator(SchemaValidator(schema), "schema")
    
    # Value range validation
    ranges = {
        'account_type': {
            'allowed_values': ['residential', 'commercial', 'industrial', 'agricultural', 'municipal']
        },
        'status': {
            'allowed_values': ['active', 'inactive', 'pending', 'closed']
        },
        'state': {
            'allowed_values': [
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
                'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
                'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
            ]
        }
    }
    runner.add_validator(ValueRangeValidator(ranges), "ranges")
    
    # Format validation
    formats = {
        'account_id': {
            'type': 'regex',
            'pattern': r'^[A-Z0-9]{6,12}$'  # Example pattern for account IDs
        },
        'email': {
            'type': 'email'
        },
        'phone': {
            'type': 'phone'
        },
        'postal_code': {
            'type': 'regex',
            'pattern': r'^\d{5}(-\d{4})?$'  # US ZIP code format
        },
        'created_date': {
            'type': 'date'
        }
    }
    runner.add_validator(FormatValidator(formats), "formats")
    
    return runner


def get_asset_data_validator() -> DataQualityRunner:
    """
    Get validator for water infrastructure asset data
    
    Returns:
        Configured DataQualityRunner
    """
    runner = DataQualityRunner()
    
    # Schema validation
    schema = {
        'columns': ['asset_id', 'asset_type', 'install_date', 'location', 'manufacturer', 
                   'model', 'capacity', 'condition', 'last_inspection', 'status'],
        'required': ['asset_id', 'asset_type', 'location', 'status'],
        'types': {
            'asset_id': 'string',
            'asset_type': 'string',
            'install_date': 'datetime64[ns]',
            'location': 'string',
            'manufacturer': 'string',
            'model': 'string',
            'capacity': 'float',
            'condition': 'string',
            'last_inspection': 'datetime64[ns]',
            'status': 'string'
        },
        'nullable': ['install_date', 'manufacturer', 'model', 'capacity', 'condition', 'last_inspection']
    }
    runner.add_validator(SchemaValidator(schema), "schema")
    
    # Value range validation
    ranges = {
        'asset_type': {
            'allowed_values': [
                'pipe', 'valve', 'pump', 'tank', 'meter', 'hydrant', 'treatment', 
                'reservoir', 'well', 'station', 'filter', 'sensor'
            ]
        },
        'condition': {
            'allowed_values': ['excellent', 'good', 'fair', 'poor', 'critical', 'unknown']
        },
        'status': {
            'allowed_values': ['active', 'inactive', 'maintenance', 'retired', 'planned']
        },
        'capacity': {
            'min': 0
        }
    }
    runner.add_validator(ValueRangeValidator(ranges), "ranges")
    
    # Complex validation
    rules = [
        {
            'name': 'inspection_date_valid',
            'description': 'Last inspection date cannot be after current date',
            'condition': "last_inspection.isna() | (last_inspection <= @pd.Timestamp('today'))",
            'columns': ['last_inspection']
        },
        {
            'name': 'install_date_valid',
            'description': 'Install date cannot be after current date',
            'condition': "install_date.isna() | (install_date <= @pd.Timestamp('today'))",
            'columns': ['install_date']
        },
        {
            'name': 'inspection_after_install',
            'description': 'Last inspection must be after installation date',
            'condition': "last_inspection.isna() | install_date.isna() | (last_inspection >= install_date)",
            'columns': ['last_inspection', 'install_date']
        }
    ]
    runner.add_validator(ComplexValidator(rules), "complex_rules")
    
    # Format validation
    formats = {
        'asset_id': {
            'type': 'regex',
            'pattern': r'^[A-Z]{2}[0-9]{6}$'  # Example pattern: 2 uppercase letters followed by 6 digits
        },
        'install_date': {
            'type': 'date'
        },
        'last_inspection': {
            'type': 'date'
        }
    }
    runner.add_validator(FormatValidator(formats), "formats")
    
    return runner


def validate_data(data: pd.DataFrame, data_type: str = 'generic') -> Dict[str, Any]:
    """
    Validate data based on its type
    
    Args:
        data: DataFrame to validate
        data_type: Type of data ('meter_reading', 'water_quality', 'customer', 'asset', or 'generic')
        
    Returns:
        Dictionary with validation results
    """
    # Select appropriate validator based on data type
    if data_type == 'meter_reading':
        runner = get_meter_reading_validator()
    elif data_type == 'water_quality':
        runner = get_water_quality_validator()
    elif data_type == 'customer':
        runner = get_customer_data_validator()
    elif data_type == 'asset':
        runner = get_asset_data_validator()
    else:
        # Generic validator for unknown data types
        runner = DataQualityRunner()
        
        # Add basic schema validation based on data
        columns = list(data.columns)
        types = {col: str(data[col].dtype) for col in columns}
        schema = {
            'columns': columns,
            'required': columns,  # Assume all columns are required
            'types': types,
            'nullable': []  # Assume no nulls allowed
        }
        runner.add_validator(SchemaValidator(schema), "auto_schema")
        
        # Add basic range validation for numeric columns
        ranges = {}
        for col in columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                ranges[col] = {
                    'min': float(data[col].min()) if not pd.isna(data[col].min()) else None,
                    'max': float(data[col].max()) if not pd.isna(data[col].max()) else None
                }
                
        if ranges:
            runner.add_validator(ValueRangeValidator(ranges), "auto_ranges")
            
        # Add date format validation for datetime columns
        formats = {}
        for col in columns:
            if pd.api.types.is_datetime64_dtype(data[col]) or 'date' in col.lower() or 'time' in col.lower():
                formats[col] = {'type': 'date'}
                
        if formats:
            runner.add_validator(FormatValidator(formats), "auto_formats")
    
    # Run validation
    results = runner.run(data)
    return results
