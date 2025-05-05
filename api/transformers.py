# Data transformation utilities for water utility data

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import json
import logging
from datetime import datetime, timedelta
import re
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """Base class for all data transformers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize transformer with configuration
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.transformation_log = []
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the provided data
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        raise NotImplementedError("Subclasses must implement transform()")
        
    def log_transformation(self, operation: str, details: Dict[str, Any]) -> None:
        """
        Log a transformation operation
        
        Args:
            operation: Name of the transformation operation
            details: Details about the transformation
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        }
        self.transformation_log.append(log_entry)
        logger.info(f"Transformation: {operation} - {json.dumps(details)}")
        
    def get_transformation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of transformation operations
        
        Returns:
            List of transformation log entries
        """
        return self.transformation_log


class ColumnRenamer(DataTransformer):
    """Transformer for renaming columns"""
    
    def __init__(self, column_map: Dict[str, str], **kwargs):
        """
        Initialize column renamer
        
        Args:
            column_map: Dictionary mapping original column names to new names
        """
        super().__init__(**kwargs)
        self.column_map = column_map
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns according to the column map
        
        Args:
            data: DataFrame to transform
            
        Returns:
            DataFrame with renamed columns
        """
        columns_before = list(data.columns)
        
        # Create a new column map with only the columns that exist in the data
        valid_column_map = {col: new_col for col, new_col in self.column_map.items() if col in data.columns}
        
        if not valid_column_map:
            # No columns to rename
            return data
            
        # Rename columns
        result = data.rename(columns=valid_column_map)
        
        # Log transformation
        self.log_transformation("rename_columns", {
            "columns_before": columns_before,
            "columns_after": list(result.columns),
            "renamed_columns": valid_column_map
        })
        
        return result


class TypeConverter(DataTransformer):
    """Transformer for converting column data types"""
    
    def __init__(self, type_map: Dict[str, str], **kwargs):
        """
        Initialize type converter
        
        Args:
            type_map: Dictionary mapping column names to desired data types
        """
        super().__init__(**kwargs)
        self.type_map = type_map
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert column data types according to the type map
        
        Args:
            data: DataFrame to transform
            
        Returns:
            DataFrame with converted data types
        """
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Track successful and failed conversions
        successful = []
        failed = []
        
        for col, dtype in self.type_map.items():
            if col not in result.columns:
                continue
                
            try:
                # Handle special case for datetime
                if dtype == 'datetime':
                    result[col] = pd.to_datetime(result[col])
                else:
                    result[col] = result[col].astype(dtype)
                
                successful.append({"column": col, "type": dtype})
            except Exception as e:
                failed.append({"column": col, "type": dtype, "error": str(e)})
                logger.warning(f"Failed to convert column {col} to {dtype}: {str(e)}")
                
        # Log transformation
        self.log_transformation("convert_types", {
            "successful": successful,
            "failed": failed
        })
        
        return result


class NullHandler(DataTransformer):
    """Transformer for handling null values"""
    
    def __init__(self, null_strategies: Dict[str, Dict[str, Any]], **kwargs):
        """
        Initialize null handler
        
        Args:
            null_strategies: Dictionary mapping column names to null handling strategies:
                - strategy: One of 'drop', 'fill', or 'flag'
                - value: Value to fill nulls with (for 'fill' strategy)
                - flag_column: Column name for flag (for 'flag' strategy)
        """
        super().__init__(**kwargs)
        self.null_strategies = null_strategies
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values according to the specified strategies
        
        Args:
            data: DataFrame to transform
            
        Returns:
            DataFrame with null values handled
        """
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Track changes
        changes = []
        
        for col, strategy_dict in self.null_strategies.items():
            if col not in result.columns:
                continue
                
            strategy = strategy_dict.get('strategy', 'fill')
            
            # Count nulls before
            null_count_before = result[col].isna().sum()
            
            if strategy == 'drop':
                # Drop rows with nulls in this column
                result = result.dropna(subset=[col])
                null_count_after = 0
            elif strategy == 'fill':
                # Fill nulls with specified value
                fill_value = strategy_dict.get('value', 0)
                result[col] = result[col].fillna(fill_value)
                null_count_after = result[col].isna().sum()
            elif strategy == 'flag':
                # Create a flag column for nulls
                flag_column = strategy_dict.get('flag_column', f"{col}_is_null")
                result[flag_column] = result[col].isna().astype(int)
                null_count_after = null_count_before
            else:
                # Unknown strategy
                logger.warning(f"Unknown null handling strategy: {strategy}")
                null_count_after = null_count_before
                
            changes.append({
                "column": col,
                "strategy": strategy,
                "null_count_before": int(null_count_before),
                "null_count_after": int(null_count_after)
            })
            
        # Log transformation
        self.log_transformation("handle_nulls", {
            "changes": changes,
            "rows_before": len(data),
            "rows_after": len(result)
        })
        
        return result


class DerivedColumnCreator(DataTransformer):
    """Transformer for creating derived columns"""
    
    def __init__(self, derived_columns: Dict[str, Dict[str, Any]], **kwargs):
        """
        Initialize derived column creator
        
        Args:
            derived_columns: Dictionary mapping new column names to creation instructions:
                - type: Type of derivation ('formula', 'aggregate', 'categorical')
                - formula: Formula to calculate value (for 'formula' type)
                - source_columns: List of source columns (for 'aggregate' type)
                - method: Aggregation method (for 'aggregate' type)
                - mapping: Value mapping (for 'categorical' type)
                - source_column: Source column (for 'categorical' type)
        """
        super().__init__(**kwargs)
        self.derived_columns = derived_columns
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived columns according to the specified instructions
        
        Args:
            data: DataFrame to transform
            
        Returns:
            DataFrame with derived columns
        """
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Track created columns
        created = []
        failed = []
        
        for new_col, instructions in self.derived_columns.items():
            try:
                col_type = instructions.get('type', 'formula')
                
                if col_type == 'formula':
                    # Create column using formula (using pandas eval)
                    formula = instructions.get('formula', '')
                    if formula:
                        result[new_col] = result.eval(formula)
                        created.append({"column": new_col, "type": "formula", "formula": formula})
                    else:
                        failed.append({"column": new_col, "type": "formula", "error": "No formula provided"})
                        
                elif col_type == 'aggregate':
                    # Create column by aggregating other columns
                    source_cols = instructions.get('source_columns', [])
                    method = instructions.get('method', 'sum')
                    
                    if not source_cols or not all(col in result.columns for col in source_cols):
                        failed.append({"column": new_col, "type": "aggregate", "error": "Missing source columns"})
                        continue
                        
                    # Select only numeric columns
                    numeric_cols = [col for col in source_cols if pd.api.types.is_numeric_dtype(result[col])]
                    
                    if not numeric_cols:
                        failed.append({"column": new_col, "type": "aggregate", "error": "No numeric source columns"})
                        continue
                        
                    # Apply aggregation method
                    if method == 'sum':
                        result[new_col] = result[numeric_cols].sum(axis=1)
                    elif method == 'avg' or method == 'mean':
                        result[new_col] = result[numeric_cols].mean(axis=1)
                    elif method == 'max':
                        result[new_col] = result[numeric_cols].max(axis=1)
                    elif method == 'min':
                        result[new_col] = result[numeric_cols].min(axis=1)
                    else:
                        failed.append({"column": new_col, "type": "aggregate", "error": f"Unknown method: {method}"})
                        continue
                        
                    created.append({
                        "column": new_col, 
                        "type": "aggregate", 
                        "method": method, 
                        "source_columns": numeric_cols
                    })
                    
                elif col_type == 'categorical':
                    # Create categorical column from mapping
                    source_col = instructions.get('source_column', '')
                    mapping = instructions.get('mapping', {})
                    
                    if not source_col or source_col not in result.columns:
                        failed.append({"column": new_col, "type": "categorical", "error": "Missing source column"})
                        continue
                        
                    if not mapping:
                        failed.append({"column": new_col, "type": "categorical", "error": "No mapping provided"})
                        continue
                        
                    # Apply mapping
                    result[new_col] = result[source_col].map(mapping)
                    
                    # Apply default value for unmapped values
                    if 'default' in instructions:
                        result[new_col] = result[new_col].fillna(instructions['default'])
                        
                    created.append({
                        "column": new_col, 
                        "type": "categorical", 
                        "source_column": source_col,
                        "mapping_size": len(mapping)
                    })
                    
                elif col_type == 'date_part':
                    # Extract part of a date
                    source_col = instructions.get('source_column', '')
                    part = instructions.get('part', 'year')
                    
                    if not source_col or source_col not in result.columns:
                        failed.append({"column": new_col, "type": "date_part", "error": "Missing source column"})
                        continue
                        
                    # Ensure source column is datetime
                    if not pd.api.types.is_datetime64_dtype(result[source_col]):
                        try:
                            date_column = pd.to_datetime(result[source_col])
                        except:
                            failed.append({"column": new_col, "type": "date_part", "error": "Source is not a valid date"})
                            continue
                    else:
                        date_column = result[source_col]
                        
                    # Extract the specified part
                    if part == 'year':
                        result[new_col] = date_column.dt.year
                    elif part == 'month':
                        result[new_col] = date_column.dt.month
                    elif part == 'day':
                        result[new_col] = date_column.dt.day
                    elif part == 'weekday':
                        result[new_col] = date_column.dt.weekday
                    elif part == 'quarter':
                        result[new_col] = date_column.dt.quarter
                    elif part == 'week':
                        result[new_col] = date_column.dt.isocalendar().week
                    else:
                        failed.append({"column": new_col, "type": "date_part", "error": f"Unknown part: {part}"})
                        continue
                        
                    created.append({
                        "column": new_col, 
                        "type": "date_part", 
                        "source_column": source_col,
                        "part": part
                    })
                    
                else:
                    # Unknown column type
                    failed.append({"column": new_col, "type": col_type, "error": "Unknown column type"})
                    
            except Exception as e:
                failed.append({"column": new_col, "error": str(e)})
                logger.warning(f"Failed to create derived column {new_col}: {str(e)}")
                
        # Log transformation
        self.log_transformation("create_derived_columns", {
            "created": created,
            "failed": failed
        })
        
        return result


class FilterTransformer(DataTransformer):
    """Transformer for filtering rows"""
    
    def __init__(self, filters: Dict[str, Dict[str, Any]], **kwargs):
        """
        Initialize filter transformer
        
        Args:
            filters: Dictionary of filter conditions:
                - column: Column to filter on
                - operator: Comparison operator ('eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'not_in', 'contains', 'between')
                - value: Value(s) to compare against
                - logic: Logic for combining with previous filter ('and' or 'or')
        """
        super().__init__(**kwargs)
        self.filters = filters
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter rows according to the specified conditions
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Filtered DataFrame
        """
        # Make a copy to avoid modifying the original
        result = data.copy()
        rows_before = len(result)
        
        # If no filters, return the original data
        if not self.filters:
            return result
            
        # Build mask for filtering
        mask = None
        applied_filters = []
        skipped_filters = []
        
        for filter_name, filter_dict in self.filters.items():
            column = filter_dict.get('column', '')
            operator = filter_dict.get('operator', 'eq')
            value = filter_dict.get('value')
            logic = filter_dict.get('logic', 'and')
            
            # Skip if column doesn't exist
            if column not in result.columns:
                skipped_filters.append({
                    "name": filter_name, 
                    "reason": f"Column '{column}' not found"
                })
                continue
                
            try:
                # Create filter mask based on operator
                filter_mask = None
                
                if operator == 'eq':
                    filter_mask = result[column] == value
                elif operator == 'ne':
                    filter_mask = result[column] != value
                elif operator == 'gt':
                    filter_mask = result[column] > value
                elif operator == 'ge':
                    filter_mask = result[column] >= value
                elif operator == 'lt':
                    filter_mask = result[column] < value
                elif operator == 'le':
                    filter_mask = result[column] <= value
                elif operator == 'in':
                    filter_mask = result[column].isin(value if isinstance(value, list) else [value])
                elif operator == 'not_in':
                    filter_mask = ~result[column].isin(value if isinstance(value, list) else [value])
                elif operator == 'contains':
                    if pd.api.types.is_string_dtype(result[column]):
                        filter_mask = result[column].str.contains(str(value), na=False)
                    else:
                        filter_mask = result[column].astype(str).str.contains(str(value), na=False)
                elif operator == 'between':
                    if isinstance(value, list) and len(value) == 2:
                        filter_mask = (result[column] >= value[0]) & (result[column] <= value[1])
                    else:
                        skipped_filters.append({
                            "name": filter_name, 
                            "reason": "Invalid value for 'between' operator"
                        })
                        continue
                elif operator == 'is_null':
                    filter_mask = result[column].isna()
                elif operator == 'is_not_null':
                    filter_mask = result[column].notna()
                else:
                    skipped_filters.append({
                        "name": filter_name, 
                        "reason": f"Unknown operator: {operator}"
                    })
                    continue
                
                # Combine with existing mask
                if mask is None:
                    mask = filter_mask
                else:
                    if logic == 'and':
                        mask = mask & filter_mask
                    elif logic == 'or':
                        mask = mask | filter_mask
                    else:
                        skipped_filters.append({
                            "name": filter_name, 
                            "reason": f"Unknown logic: {logic}"
                        })
                        continue
                
                applied_filters.append({
                    "name": filter_name, 
                    "column": column, 
                    "operator": operator, 
                    "logic": logic
                })
                
            except Exception as e:
                skipped_filters.append({
                    "name": filter_name, 
                    "reason": str(e)
                })
                logger.warning(f"Failed to apply filter {filter_name}: {str(e)}")
                
        # Apply mask if any filters were applied
        if mask is not None:
            result = result.loc[mask]
            
        rows_after = len(result)
        
        # Log transformation
        self.log_transformation("filter_rows", {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_filtered": rows_before - rows_after,
            "applied_filters": applied_filters,
            "skipped_filters": skipped_filters
        })
        
        return result


class AggregationTransformer(DataTransformer):
    """Transformer for aggregating data"""
    
    def __init__(self, aggregation: Dict[str, Any], **kwargs):
        """
        Initialize aggregation transformer
        
        Args:
            aggregation: Aggregation configuration:
                - group_by: Columns to group by
                - aggregations: Dictionary mapping columns to aggregation methods
        """
        super().__init__(**kwargs)
        self.aggregation = aggregation
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data according to the specified configuration
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Aggregated DataFrame
        """
        # Get aggregation configuration
        group_by = self.aggregation.get('group_by', [])
        aggregations = self.aggregation.get('aggregations', {})
        
        # If no group_by or aggregations, return the original data
        if not group_by or not aggregations:
            return data
            
        # Check if all group_by columns exist
        missing_columns = [col for col in group_by if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing group_by columns: {missing_columns}")
            # Filter out missing columns
            group_by = [col for col in group_by if col in data.columns]
            if not group_by:
                return data
                
        # Filter aggregations to include only existing columns
        valid_aggregations = {}
        for col, agg in aggregations.items():
            if col in data.columns:
                valid_aggregations[col] = agg
                
        if not valid_aggregations:
            return data
            
        # Perform aggregation
        try:
            result = data.groupby(group_by).agg(valid_aggregations).reset_index()
            
            # Log transformation
            self.log_transformation("aggregate_data", {
                "group_by": group_by,
                "aggregations": valid_aggregations,
                "rows_before": len(data),
                "rows_after": len(result)
            })
            
            return result
        except Exception as e:
            logger.warning(f"Failed to aggregate data: {str(e)}")
            return data


class TimeseriesTransformer(DataTransformer):
    """Transformer for time series data"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize time series transformer
        
        Args:
            config: Time series configuration:
                - time_column: Name of timestamp column
                - value_column: Name of value column
                - operations: List of operations to perform
                - frequency: Target frequency for resampling
        """
        super().__init__(config, **kwargs)
        self.time_column = config.get('time_column', 'timestamp')
        self.value_column = config.get('value_column', 'value')
        self.operations = config.get('operations', [])
        self.frequency = config.get('frequency', 'D')  # Default to daily
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform time series data
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        # Check if required columns exist
        if self.time_column not in data.columns:
            logger.warning(f"Time column '{self.time_column}' not found")
            return data
            
        if self.value_column not in data.columns:
            logger.warning(f"Value column '{self.value_column}' not found")
            return data
            
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(result[self.time_column]):
            try:
                result[self.time_column] = pd.to_datetime(result[self.time_column])
            except Exception as e:
                logger.warning(f"Failed to convert time column to datetime: {str(e)}")
                return data
                
        operations_performed = []
        
        # Apply operations
        for operation in self.operations:
            op_type = operation.get('type', '')
            
            try:
                if op_type == 'resample':
                    # Resample time series to a different frequency
                    freq = operation.get('frequency', self.frequency)
                    method = operation.get('method', 'mean')
                    
                    # Set time column as index
                    result = result.set_index(self.time_column)
                    
                    # Resample
                    if method == 'mean':
                        result = result.resample(freq).mean().reset_index()
                    elif method == 'sum':
                        result = result.resample(freq).sum().reset_index()
                    elif method == 'min':
                        result = result.resample(freq).min().reset_index()
                    elif method == 'max':
                        result = result.resample(freq).max().reset_index()
                    elif method == 'first':
                        result = result.resample(freq).first().reset_index()
                    elif method == 'last':
                        result = result.resample(freq).last().reset_index()
                    elif method == 'count':
                        result = result.resample(freq).count().reset_index()
                    else:
                        logger.warning(f"Unknown resample method: {method}")
                        continue
                        
                    operations_performed.append({
                        "type": "resample", 
                        "frequency": freq, 
                        "method": method
                    })
                    
                elif op_type == 'rolling':
                    # Calculate rolling statistics
                    window = operation.get('window', 7)
                    method = operation.get('method', 'mean')
                    new_column = operation.get('new_column', f"{self.value_column}_{method}_{window}")
                    
                    # Calculate rolling statistic
                    if method == 'mean':
                        result[new_column] = result[self.value_column].rolling(window=window).mean()
                    elif method == 'sum':
                        result[new_column] = result[self.value_column].rolling(window=window).sum()
                    elif method == 'min':
                        result[new_column] = result[self.value_column].rolling(window=window).min()
                    elif method == 'max':
                        result[new_column] = result[self.value_column].rolling(window=window).max()
                    elif method == 'std':
                        result[new_column] = result[self.value_column].rolling(window=window).std()
                    else:
                        logger.warning(f"Unknown rolling method: {method}")
                        continue
                        
                    operations_performed.append({
                        "type": "rolling", 
                        "window": window, 
                        "method": method, 
                        "new_column": new_column
                    })
                    
                elif op_type == 'diff':
                    # Calculate difference between consecutive periods
                    periods = operation.get('periods', 1)
                    new_column = operation.get('new_column', f"{self.value_column}_diff_{periods}")
                    
                    result[new_column] = result[self.value_column].diff(periods=periods)
                    
                    operations_performed.append({
                        "type": "diff", 
                        "periods": periods, 
                        "new_column": new_column
                    })
                    
                elif op_type == 'pct_change':
                    # Calculate percentage change between consecutive periods
                    periods = operation.get('periods', 1)
                    new_column = operation.get('new_column', f"{self.value_column}_pct_{periods}")
                    
                    result[new_column] = result[self.value_column].pct_change(periods=periods)
                    
                    operations_performed.append({
                        "type": "pct_change", 
                        "periods": periods, 
                        "new_column": new_column
                    })
                    
                elif op_type == 'shift':
                    # Shift values backward or forward
                    periods = operation.get('periods', 1)
                    new_column = operation.get('new_column', f"{self.value_column}_shift_{periods}")
                    
                    result[new_column] = result[self.value_column].shift(periods=periods)
                    
                    operations_performed.append({
                        "type": "shift", 
                        "periods": periods, 
                        "new_column": new_column
                    })
                    
                elif op_type == 'ewm':
                    # Exponentially weighted moving average
                    alpha = operation.get('alpha', 0.3)
                    new_column = operation.get('new_column', f"{self.value_column}_ewm_{alpha}")
                    
                    result[new_column] = result[self.value_column].ewm(alpha=alpha).mean()
                    
                    operations_performed.append({
                        "type": "ewm", 
                        "alpha": alpha, 
                        "new_column": new_column
                    })
                    
                else:
                    logger.warning(f"Unknown time series operation: {op_type}")
                    
            except Exception as e:
                logger.warning(f"Failed to apply time series operation {op_type}: {str(e)}")
                
        # Log transformation
        self.log_transformation("time_series_transform", {
            "operations_performed": operations_performed,
            "rows_before": len(data),
            "rows_after": len(result)
        })
        
        return result


class OutlierHandler(DataTransformer):
    """Transformer for handling outliers"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize outlier handler
        
        Args:
            config: Outlier handling configuration:
                - columns: List of columns to check for outliers
                - method: Method for detecting outliers ('zscore', 'iqr', 'percentile')
                - threshold: Threshold for outlier detection
                - strategy: Strategy for handling outliers ('remove', 'cap', 'flag')
        """
        super().__init__(config, **kwargs)
        self.columns = config.get('columns', [])
        self.method = config.get('method', 'zscore')
        self.threshold = config.get('threshold', 3.0)
        self.strategy = config.get('strategy', 'flag')
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in the data
        
        Args:
            data: DataFrame to transform
            
        Returns:
            DataFrame with outliers handled
        """
        # If no columns specified, return the original data
        if not self.columns:
            return data
            
        # Make a copy to avoid modifying the original
        result = data.copy()
        rows_before = len(result)
        
        # Filter columns to include only existing numeric columns
        valid_columns = []
        for col in self.columns:
            if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                valid_columns.append(col)
                
        if not valid_columns:
            return result
            
        outliers_detected = {}
        
        # Detect and handle outliers for each column
        for col in valid_columns:
            # Skip columns with all nulls
            if result[col].isna().all():
                continue
                
            # Detect outliers
            outlier_mask = None
            
            if self.method == 'zscore':
                # Z-score method
                zscore = (result[col] - result[col].mean()) / result[col].std()
                outlier_mask = abs(zscore) > self.threshold
                
            elif self.method == 'iqr':
                # IQR method
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.threshold * iqr
                upper_bound = q3 + self.threshold * iqr
                outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
                
            elif self.method == 'percentile':
                # Percentile method
                lower_bound = result[col].quantile(self.threshold / 100)
                upper_bound = result[col].quantile(1 - self.threshold / 100)
                outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
                
            else:
                logger.warning(f"Unknown outlier detection method: {self.method}")
                continue
                
            # Count outliers
            outlier_count = outlier_mask.sum()
            outliers_detected[col] = int(outlier_count)
            
            # Handle outliers
            if outlier_count > 0:
                if self.strategy == 'remove':
                    # Remove rows with outliers
                    result = result[~outlier_mask]
                    
                elif self.strategy == 'cap':
                    # Cap outliers at threshold values
                    if self.method == 'zscore':
                        mean = result[col].mean()
                        std = result[col].std()
                        result.loc[outlier_mask & (result[col] > mean), col] = mean + self.threshold * std
                        result.loc[outlier_mask & (result[col] < mean), col] = mean - self.threshold * std
                    elif self.method == 'iqr':
                        q1 = result[col].quantile(0.25)
                        q3 = result[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - self.threshold * iqr
                        upper_bound = q3 + self.threshold * iqr
                        result.loc[result[col] < lower_bound, col] = lower_bound
                        result.loc[result[col] > upper_bound, col] = upper_bound
                    elif self.method == 'percentile':
                        lower_bound = result[col].quantile(self.threshold / 100)
                        upper_bound = result[col].quantile(1 - self.threshold / 100)
                        result.loc[result[col] < lower_bound, col] = lower_bound
                        result.loc[result[col] > upper_bound, col] = upper_bound
                        
                elif self.strategy == 'flag':
                    # Flag outliers with a new column
                    result[f"{col}_outlier"] = outlier_mask.astype(int)
                    
                else:
                    logger.warning(f"Unknown outlier handling strategy: {self.strategy}")
                    
        rows_after = len(result)
        
        # Log transformation
        self.log_transformation("handle_outliers", {
            "method": self.method,
            "strategy": self.strategy,
            "threshold": self.threshold,
            "outliers_detected": outliers_detected,
            "rows_before": rows_before,
            "rows_after": rows_after
        })
        
        return result


class TransformationPipeline:
    """Pipeline for applying multiple transformations"""
    
    def __init__(self, name: str = "default_pipeline"):
        """
        Initialize transformation pipeline
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.transformers = []
        self.execution_log = []
        
    def add_transformer(self, transformer: DataTransformer, name: Optional[str] = None) -> None:
        """
        Add a transformer to the pipeline
        
        Args:
            transformer: Transformer to add
            name: Optional name for the transformer
        """
        if name is None:
            name = f"transformer_{len(self.transformers)}"
            
        self.transformers.append((name, transformer))
        
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the transformation pipeline
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        start_time = datetime.now()
        execution_details = []
        
        for name, transformer in self.transformers:
            try:
                transformer_start = datetime.now()
                
                # Apply transformation
                result = transformer.transform(result)
                
                transformer_end = datetime.now()
                duration = (transformer_end - transformer_start).total_seconds()
                
                # Record execution details
                execution_details.append({
                    "transformer": name,
                    "start_time": transformer_start.isoformat(),
                    "end_time": transformer_end.isoformat(),
                    "duration_seconds": duration,
                    "rows_before": len(data),
                    "rows_after": len(result),
                    "columns_before": list(data.columns),
                    "columns_after": list(result.columns),
                    "transformation_log": transformer.transformation_log
                })
                
            except Exception as e:
                logger.error(f"Error executing transformer {name}: {str(e)}")
                execution_details.append({
                    "transformer": name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Record overall execution
        execution_log = {
            "pipeline": self.name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": total_duration,
            "transformers_executed": len(self.transformers),
            "rows_before": len(data),
            "rows_after": len(result),
            "columns_before": list(data.columns),
            "columns_after": list(result.columns),
            "execution_details": execution_details
        }
        
        self.execution_log.append(execution_log)
        
        return result
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of pipeline executions
        
        Returns:
            List of execution log entries
        """
        return self.execution_log
    
    def get_last_execution(self) -> Optional[Dict[str, Any]]:
        """
        Get the details of the last pipeline execution
        
        Returns:
            Execution log entry for the last execution, or None if no executions
        """
        if self.execution_log:
            return self.execution_log[-1]
        return None


# Client-specific transformers for common water utility data formats

def create_thames_water_pipeline() -> TransformationPipeline:
    """
    Create a transformation pipeline for Thames Water data format
    
    Returns:
        Configured TransformationPipeline
    """
    pipeline = TransformationPipeline(name="thames_water_pipeline")
    
    # Standard column renaming
    column_map = {
        'Site_ID': 'site_id',
        'Site_Name': 'site_name',
        'Date': 'date',
        'DateTime': 'datetime',
        'Consumption': 'consumption',
        'Consumption_m3': 'consumption_m3',
        'Meter_ID': 'meter_id',
        'Meter_Reading': 'meter_reading',
        'Catchment_Area': 'catchment_area',
        'Programme': 'programme',
        'Status': 'status',
        'Budget': 'budget',
        'Actual': 'actual',
        'Forecast': 'forecast',
        'Variance': 'variance'
    }
    pipeline.add_transformer(ColumnRenamer(column_map), "standard_renaming")
    
    # Type conversion
    type_map = {
        'date': 'datetime',
        'datetime': 'datetime',
        'consumption': 'float',
        'consumption_m3': 'float',
        'meter_reading': 'float',
        'budget': 'float',
        'actual': 'float',
        'forecast': 'float',
        'variance': 'float'
    }
    pipeline.add_transformer(TypeConverter(type_map), "type_conversion")
    
    # Handle nulls
    null_strategies = {
        'consumption': {'strategy': 'fill', 'value': 0},
        'consumption_m3': {'strategy': 'fill', 'value': 0},
        'meter_reading': {'strategy': 'fill', 'value': 0},
        'budget': {'strategy': 'fill', 'value': 0},
        'actual': {'strategy': 'fill', 'value': 0},
        'forecast': {'strategy': 'fill', 'value': 0},
        'variance': {'strategy': 'fill', 'value': 0}
    }
    pipeline.add_transformer(NullHandler(null_strategies), "handle_nulls")
    
    # Derived columns
    derived_columns = {
        'budget_variance_pct': {
            'type': 'formula',
            'formula': '(actual / budget - 1) * 100 if budget > 0 else 0'
        },
        'year': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'year'
        },
        'month': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'month'
        },
        'quarter': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'quarter'
        }
    }
    pipeline.add_transformer(DerivedColumnCreator(derived_columns), "create_derived_columns")
    
    return pipeline


def create_uk_water_company_pipeline() -> TransformationPipeline:
    """
    Create a transformation pipeline for generic UK water company data
    
    Returns:
        Configured TransformationPipeline
    """
    pipeline = TransformationPipeline(name="uk_water_company_pipeline")
    
    # Column renaming
    column_map = {
        'SITE_ID': 'site_id',
        'SITE_NAME': 'site_name',
        'SITE_TYPE': 'site_type',
        'DATE': 'date',
        'DATE_TIME': 'datetime',
        'CONSUMPTION': 'consumption',
        'METER_ID': 'meter_id',
        'METER_TYPE': 'meter_type',
        'METER_READING': 'meter_reading',
        'UNIT': 'unit',
        'REGION': 'region',
        'CATCHMENT': 'catchment_area',
        'PROGRAMME': 'programme',
        'STATUS': 'status',
        'BUDGET': 'budget',
        'ACTUAL': 'actual',
        'FORECAST': 'forecast',
        'VARIANCE': 'variance'
    }
    pipeline.add_transformer(ColumnRenamer(column_map), "column_renaming")
    
    # Type conversion
    type_map = {
        'date': 'datetime',
        'datetime': 'datetime',
        'consumption': 'float',
        'meter_reading': 'float',
        'budget': 'float',
        'actual': 'float',
        'forecast': 'float',
        'variance': 'float'
    }
    pipeline.add_transformer(TypeConverter(type_map), "type_conversion")
    
    # Unit conversion - convert to standard units if needed
    # Assuming a derived column that standardizes measurements
    derived_columns = {
        'consumption_m3': {
            'type': 'formula',
            'formula': "consumption * 0.0283168 if unit == 'cf' else (consumption * 0.001 if unit == 'liters' else consumption)"
        },
        'year': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'year'
        },
        'month': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'month'
        },
        'budget_variance_pct': {
            'type': 'formula',
            'formula': '(actual / budget - 1) * 100 if budget > 0 else 0'
        }
    }
    pipeline.add_transformer(DerivedColumnCreator(derived_columns), "derived_columns")
    
    # Handle nulls
    null_strategies = {
        'consumption': {'strategy': 'fill', 'value': 0},
        'meter_reading': {'strategy': 'fill', 'value': 0},
        'budget': {'strategy': 'fill', 'value': 0},
        'actual': {'strategy': 'fill', 'value': 0},
        'forecast': {'strategy': 'fill', 'value': 0},
        'variance': {'strategy': 'fill', 'value': 0}
    }
    pipeline.add_transformer(NullHandler(null_strategies), "null_handler")
    
    return pipeline


def create_us_water_utility_pipeline() -> TransformationPipeline:
    """
    Create a transformation pipeline for US water utility data format
    
    Returns:
        Configured TransformationPipeline
    """
    pipeline = TransformationPipeline(name="us_water_utility_pipeline")
    
    # Column renaming
    column_map = {
        'FacilityID': 'site_id',
        'FacilityName': 'site_name',
        'FacilityType': 'site_type',
        'ReadDate': 'date',
        'ReadDateTime': 'datetime',
        'Usage': 'consumption',
        'UsageGallons': 'consumption_gallons',
        'MeterID': 'meter_id',
        'MeterType': 'meter_type',
        'ReadValue': 'meter_reading',
        'District': 'district',
        'Zone': 'zone',
        'BudgetAmount': 'budget',
        'ActualAmount': 'actual',
        'ProjectedAmount': 'forecast',
        'VarianceAmount': 'variance'
    }
    pipeline.add_transformer(ColumnRenamer(column_map), "column_renaming")
    
    # Type conversion
    type_map = {
        'date': 'datetime',
        'datetime': 'datetime',
        'consumption': 'float',
        'consumption_gallons': 'float',
        'meter_reading': 'float',
        'budget': 'float',
        'actual': 'float',
        'forecast': 'float',
        'variance': 'float'
    }
    pipeline.add_transformer(TypeConverter(type_map), "type_conversion")
    
    # Unit conversion - convert to standard units if needed
    derived_columns = {
        'consumption_m3': {
            'type': 'formula',
            'formula': 'consumption_gallons * 0.00378541'  # Convert gallons to cubic meters
        },
        'year': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'year'
        },
        'month': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'month'
        },
        'quarter': {
            'type': 'date_part',
            'source_column': 'date',
            'part': 'quarter'
        },
        'budget_variance_pct': {
            'type': 'formula',
            'formula': '(actual / budget - 1) * 100 if budget > 0 else 0'
        }
    }
    pipeline.add_transformer(DerivedColumnCreator(derived_columns), "derived_columns")
    
    # Handle nulls
    null_strategies = {
        'consumption': {'strategy': 'fill', 'value': 0},
        'consumption_gallons': {'strategy': 'fill', 'value': 0},
        'meter_reading': {'strategy': 'fill', 'value': 0},
        'budget': {'strategy': 'fill', 'value': 0},
        'actual': {'strategy': 'fill', 'value': 0},
        'forecast': {'strategy': 'fill', 'value': 0},
        'variance': {'strategy': 'fill', 'value': 0}
    }
    pipeline.add_transformer(NullHandler(null_strategies), "null_handler")
    
    return pipeline


def transform_data(data: pd.DataFrame, client_format: str = 'thames', custom_pipeline: Optional[TransformationPipeline] = None) -> pd.DataFrame:
    """
    Transform data based on client format
    
    Args:
        data: DataFrame to transform
        client_format: Client data format ('thames', 'uk_water', 'us_water', or 'custom')
        custom_pipeline: Optional custom transformation pipeline
        
    Returns:
        Transformed DataFrame
    """
    if custom_pipeline is not None:
        pipeline = custom_pipeline
    elif client_format == 'thames':
        pipeline = create_thames_water_pipeline()
    elif client_format == 'uk_water':
        pipeline = create_uk_water_company_pipeline()
    elif client_format == 'us_water':
        pipeline = create_us_water_utility_pipeline()
    else:
        # Generic pipeline
        pipeline = TransformationPipeline(name="generic_pipeline")
        
        # Add basic transformers
        column_map = {col: col.lower().replace(' ', '_') for col in data.columns}
        pipeline.add_transformer(ColumnRenamer(column_map), "lowercase_columns")
        
    # Execute pipeline
    result = pipeline.execute(data)
    return result

