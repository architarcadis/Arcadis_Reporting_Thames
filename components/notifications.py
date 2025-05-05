# Notification and Alert Prioritization Component

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alert severity levels
SEVERITY_LEVELS = {
    'critical': {
        'level': 1,
        'description': 'Requires immediate attention, significant impact on service',
        'color': '#FF4B4B'  # Red
    },
    'high': {
        'level': 2,
        'description': 'Requires urgent attention, potential service impact',
        'color': '#FFB107'  # Amber
    },
    'medium': {
        'level': 3,
        'description': 'Should be addressed soon, minor service impact',
        'color': '#FFD700'  # Yellow
    },
    'low': {
        'level': 4,
        'description': 'Can be addressed during normal operations',
        'color': '#28A745'  # Green
    },
    'info': {
        'level': 5,
        'description': 'Informational, no service impact',
        'color': '#00A1D6'  # Light blue
    }
}

# Default prioritization configuration
DEFAULT_PRIORITY_CONFIG = {
    'severity_weights': {
        'critical': 100,
        'high': 80,
        'medium': 50,
        'low': 20,
        'info': 5
    },
    'factor_weights': {
        'customer_impact': 3.0,
        'regulatory': 2.5,
        'financial': 2.0,
        'operational': 1.5,
        'reputational': 1.0,
        'safety': 4.0,
        'environmental': 2.0
    },
    'time_decay': {
        'enabled': True,
        'half_life_days': 14,  # Priority halves every 14 days
        'min_factor': 0.25     # Priority never decays below 25% of original
    },
    'max_active_alerts': 25    # Maximum number of active alerts to track
}

def calculate_alert_priority(alert: Dict[str, Any], config: Dict[str, Any] = None) -> float:
    """Calculate a numeric priority score for an alert based on various factors
    
    Args:
        alert: Alert data dictionary containing severity, factors, etc.
        config: Custom prioritization configuration (optional)
        
    Returns:
        Numeric priority score (higher = higher priority)
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_PRIORITY_CONFIG
        
    severity = alert.get('severity', 'medium').lower()
    created_at = alert.get('created_at', datetime.now().isoformat())
    
    # Get base priority from severity
    severity_weight = config['severity_weights'].get(severity, config['severity_weights']['medium'])
    priority = severity_weight
    
    # Apply factor multipliers
    factors = alert.get('factors', {})
    for factor, value in factors.items():
        if factor in config['factor_weights'] and isinstance(value, (int, float)) and value > 0:
            factor_weight = config['factor_weights'][factor]
            priority *= (1 + (value * factor_weight / 100))
    
    # Apply time decay if enabled
    if config['time_decay']['enabled']:
        try:
            # Parse time strings
            if isinstance(created_at, str):
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_dt = created_at
                
            # Calculate age in days
            age_days = (datetime.now() - created_dt).total_seconds() / (24 * 3600)
            
            # Apply exponential decay with half-life
            half_life = config['time_decay']['half_life_days']
            decay_factor = max(config['time_decay']['min_factor'], 2 ** (-age_days / half_life))
            priority *= decay_factor
        except Exception as e:
            logger.warning(f"Error calculating time decay: {str(e)}")
    
    # Apply acknowledgement reduction
    if alert.get('acknowledged', False):
        priority *= 0.8  # Reduce priority by 20% if acknowledged
    
    # Apply resolution workflow adjustment
    if 'resolution_workflow' in alert:
        if alert['resolution_workflow'].get('started', False):
            priority *= 0.7  # Reduce priority by 30% if resolution started
            
            # Further reduce based on progress
            progress = alert['resolution_workflow'].get('progress', 0)
            if isinstance(progress, (int, float)) and 0 <= progress <= 100:
                priority *= (1 - (progress / 200))  # Max 50% reduction at 100% progress
    
    return priority

def prioritize_alerts(alerts: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Prioritize a list of alerts based on configured rules
    
    Args:
        alerts: List of alert dictionaries
        config: Custom prioritization configuration (optional)
        
    Returns:
        List of alerts with priority scores, sorted by priority
    """
    if not alerts:
        return []
        
    # Use default config if none provided
    if config is None:
        config = DEFAULT_PRIORITY_CONFIG
        
    # Calculate priority for each alert
    for alert in alerts:
        # Skip alerts already resolved
        if alert.get('status') == 'resolved':
            alert['priority_score'] = 0
            continue
            
        alert['priority_score'] = calculate_alert_priority(alert, config)
        
        # Add additional context
        alert['priority_level'] = get_priority_level(alert['priority_score'])
    
    # Sort by priority (highest first)
    sorted_alerts = sorted(alerts, key=lambda x: x.get('priority_score', 0), reverse=True)
    
    # Limit to max active alerts if specified
    max_alerts = config.get('max_active_alerts', 0)
    if max_alerts > 0:
        active_alerts = [a for a in sorted_alerts if a.get('status') != 'resolved']
        if len(active_alerts) > max_alerts:
            # Mark excess low-priority alerts as auto-archived
            for alert in active_alerts[max_alerts:]:
                alert['status'] = 'archived'
                alert['auto_archived'] = True
                alert['archived_reason'] = 'Exceeded maximum active alerts threshold'
    
    return sorted_alerts

def get_priority_level(score: float) -> str:
    """Convert a numeric priority score to a descriptive level
    
    Args:
        score: Numeric priority score
        
    Returns:
        Priority level string
    """
    if score >= 150:
        return 'critical'
    elif score >= 100:
        return 'high'
    elif score >= 50:
        return 'medium'
    elif score >= 20:
        return 'low'
    else:
        return 'info'

def generate_alert(alert_type: str, source: str, message: str, 
                  severity: str = 'medium', tenant_id: str = None,
                  details: Dict[str, Any] = None, factors: Dict[str, float] = None) -> Dict[str, Any]:
    """Generate a structured alert object
    
    Args:
        alert_type: Type of alert (e.g., 'sensor', 'compliance', 'system')
        source: Source system or component generating the alert
        message: Alert message text
        severity: Alert severity level
        tenant_id: Tenant identifier (optional)
        details: Additional alert details (optional)
        factors: Impact factors for prioritization (optional)
        
    Returns:
        Alert dictionary
    """
    alert_id = f"{alert_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(message) % 10000}"
    
    alert = {
        'id': alert_id,
        'type': alert_type,
        'source': source,
        'message': message,
        'severity': severity.lower(),
        'created_at': datetime.now().isoformat(),
        'status': 'active',
        'acknowledged': False
    }
    
    if tenant_id:
        alert['tenant_id'] = tenant_id
        
    if details:
        alert['details'] = details
        
    if factors:
        alert['factors'] = factors
    
    # Calculate initial priority
    alert['priority_score'] = calculate_alert_priority(alert)
    alert['priority_level'] = get_priority_level(alert['priority_score'])
    
    return alert

# Main function to be called from the application
def prioritize_alerts(alerts: List[Dict[str, Any]], tenant_id: str = None, 
                     custom_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Prioritize alerts for display and handling
    
    Args:
        alerts: List of alert dictionaries
        tenant_id: Tenant identifier (optional)
        custom_config: Custom prioritization configuration (optional)
        
    Returns:
        Prioritized list of alerts
    """
    # Customize configuration based on tenant if needed
    config = DEFAULT_PRIORITY_CONFIG.copy()
    
    if custom_config:
        # Deep merge the configurations
        for key, value in custom_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Merge nested dictionaries
                config[key] = {**config[key], **value}
            else:
                # Replace top-level keys
                config[key] = value
    
    # Filter alerts by tenant if specified
    if tenant_id:
        tenant_alerts = [a for a in alerts if a.get('tenant_id') == tenant_id or not a.get('tenant_id')]
    else:
        tenant_alerts = alerts
    
    return prioritize_alerts(tenant_alerts, config)

# For testing
if __name__ == "__main__":
    # Generate sample alerts
    test_alerts = [
        generate_alert(
            alert_type="sensor",
            source="pressure_sensor_123",
            message="High pressure detected in main supply line",
            severity="high",
            tenant_id="test_tenant",
            details={
                "location": "Main Pump Station",
                "reading": 152,
                "threshold": 130,
                "unit": "PSI"
            },
            factors={
                "operational": 0.8,
                "safety": 0.5
            }
        ),
        generate_alert(
            alert_type="compliance",
            source="water_quality_monitor",
            message="pH levels outside regulatory range",
            severity="critical",
            tenant_id="test_tenant",
            details={
                "location": "Treatment Plant A",
                "reading": 9.2,
                "threshold_min": 6.5,
                "threshold_max": 8.5,
                "unit": "pH"
            },
            factors={
                "regulatory": 0.9,
                "environmental": 0.7,
                "reputational": 0.6
            }
        ),
        generate_alert(
            alert_type="system",
            source="billing_system",
            message="Failed to generate monthly billing report",
            severity="medium",
            tenant_id="test_tenant",
            details={
                "component": "Report Generator",
                "error": "Database timeout",
                "affected_customers": 0
            },
            factors={
                "operational": 0.4,
                "financial": 0.2
            }
        )
    ]
    
    # Add an older alert with time decay
    old_alert = generate_alert(
        alert_type="maintenance",
        source="pump_station_2",
        message="Scheduled maintenance due",
        severity="low",
        tenant_id="test_tenant",
        details={
            "asset_id": "PS2-PUMP-03",
            "maintenance_type": "Quarterly Inspection"
        }
    )
    # Backdate the alert by 20 days
    old_alert['created_at'] = (datetime.now() - timedelta(days=20)).isoformat()
    test_alerts.append(old_alert)
    
    # Prioritize and print results
    prioritized = prioritize_alerts(test_alerts, "test_tenant")
    
    print("Alert Priority Ranking:")
    for i, alert in enumerate(prioritized):
        print(f"{i+1}. [{alert['severity'].upper()}] {alert['message']} (Score: {alert['priority_score']:.2f})")
