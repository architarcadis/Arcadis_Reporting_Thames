# Financial Impact and ROI Calculator Component

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialImpactCalculator:
    """Class to calculate financial impacts and ROI of operational changes"""
    
    def __init__(self, tenant_id: str, currency: str = "£"):
        """Initialize with tenant information and currency"""
        self.tenant_id = tenant_id
        self.currency = currency
        self.assumptions = {}
        self.results = {}
        
    def set_assumptions(self, assumptions: Dict[str, Any]) -> None:
        """Set financial model assumptions"""
        self.assumptions = assumptions
        
    def calculate_water_loss_impact(self, 
                                   current_loss_pct: float, 
                                   target_loss_pct: float, 
                                   annual_water_volume: float,
                                   unit_cost: float,
                                   implementation_cost: float = 0,
                                   timeframe_years: int = 1) -> Dict[str, Any]:
        """Calculate financial impact of reducing water losses
        
        Args:
            current_loss_pct: Current water loss percentage
            target_loss_pct: Target water loss percentage after improvements
            annual_water_volume: Total annual water volume in cubic meters
            unit_cost: Cost per cubic meter of water
            implementation_cost: Cost to implement loss reduction measures
            timeframe_years: Years over which to calculate impact
            
        Returns:
            Dictionary with financial impact results
        """
        try:
            # Calculate reduced loss volume
            current_loss_volume = annual_water_volume * (current_loss_pct / 100)
            target_loss_volume = annual_water_volume * (target_loss_pct / 100)
            reduced_loss_volume = current_loss_volume - target_loss_volume
            
            # Calculate annual savings
            annual_savings = reduced_loss_volume * unit_cost
            total_savings = annual_savings * timeframe_years
            
            # Calculate ROI
            if implementation_cost > 0:
                roi_pct = ((total_savings - implementation_cost) / implementation_cost) * 100
                payback_period = implementation_cost / annual_savings if annual_savings > 0 else float('inf')
            else:
                roi_pct = float('inf')
                payback_period = 0
            
            result = {
                "metric": "water_loss",
                "current_state": {
                    "loss_percentage": current_loss_pct,
                    "loss_volume": current_loss_volume,
                    "loss_cost": current_loss_volume * unit_cost
                },
                "target_state": {
                    "loss_percentage": target_loss_pct,
                    "loss_volume": target_loss_volume,
                    "loss_cost": target_loss_volume * unit_cost
                },
                "impact": {
                    "reduced_loss_volume": reduced_loss_volume,
                    "annual_savings": annual_savings,
                    "total_savings": total_savings,
                    "implementation_cost": implementation_cost,
                    "net_benefit": total_savings - implementation_cost,
                    "roi_percentage": roi_pct,
                    "payback_period_years": payback_period
                },
                "assumptions": {
                    "timeframe_years": timeframe_years,
                    "unit_cost": unit_cost,
                    "annual_water_volume": annual_water_volume,
                    "currency": self.currency
                }
            }
            
            self.results["water_loss"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error calculating water loss impact: {str(e)}")
            return {"error": f"Water loss calculation failed: {str(e)}"}
    
    def calculate_energy_efficiency_impact(self,
                                          current_energy_usage: float,
                                          target_energy_usage: float,
                                          energy_unit_cost: float,
                                          carbon_factor: float = 0.233,  # kg CO2e per kWh
                                          carbon_price: float = 0,  # Carbon price per ton
                                          implementation_cost: float = 0,
                                          timeframe_years: int = 1) -> Dict[str, Any]:
        """Calculate financial impact of improving energy efficiency
        
        Args:
            current_energy_usage: Current annual energy usage in kWh
            target_energy_usage: Target annual energy usage after improvements in kWh
            energy_unit_cost: Cost per kWh
            carbon_factor: Carbon emission factor (kg CO2e per kWh)
            carbon_price: Price per ton of carbon emissions
            implementation_cost: Cost to implement efficiency measures
            timeframe_years: Years over which to calculate impact
            
        Returns:
            Dictionary with financial impact results
        """
        try:
            # Calculate energy savings
            energy_savings = current_energy_usage - target_energy_usage
            annual_cost_savings = energy_savings * energy_unit_cost
            
            # Calculate carbon reduction
            carbon_reduction_kg = energy_savings * carbon_factor
            carbon_reduction_tons = carbon_reduction_kg / 1000
            carbon_value = carbon_reduction_tons * carbon_price
            
            # Calculate total savings
            annual_total_savings = annual_cost_savings + carbon_value
            total_savings = annual_total_savings * timeframe_years
            
            # Calculate ROI
            if implementation_cost > 0:
                roi_pct = ((total_savings - implementation_cost) / implementation_cost) * 100
                payback_period = implementation_cost / annual_total_savings if annual_total_savings > 0 else float('inf')
            else:
                roi_pct = float('inf')
                payback_period = 0
            
            result = {
                "metric": "energy_efficiency",
                "current_state": {
                    "energy_usage": current_energy_usage,
                    "energy_cost": current_energy_usage * energy_unit_cost,
                    "carbon_emissions_tons": (current_energy_usage * carbon_factor) / 1000
                },
                "target_state": {
                    "energy_usage": target_energy_usage,
                    "energy_cost": target_energy_usage * energy_unit_cost,
                    "carbon_emissions_tons": (target_energy_usage * carbon_factor) / 1000
                },
                "impact": {
                    "energy_savings": energy_savings,
                    "annual_cost_savings": annual_cost_savings,
                    "carbon_reduction_tons": carbon_reduction_tons,
                    "carbon_value": carbon_value,
                    "annual_total_savings": annual_total_savings,
                    "total_savings": total_savings,
                    "implementation_cost": implementation_cost,
                    "net_benefit": total_savings - implementation_cost,
                    "roi_percentage": roi_pct,
                    "payback_period_years": payback_period
                },
                "assumptions": {
                    "timeframe_years": timeframe_years,
                    "energy_unit_cost": energy_unit_cost,
                    "carbon_factor": carbon_factor,
                    "carbon_price": carbon_price,
                    "currency": self.currency
                }
            }
            
            self.results["energy_efficiency"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error calculating energy efficiency impact: {str(e)}")
            return {"error": f"Energy efficiency calculation failed: {str(e)}"}
    
    def calculate_compliance_impact(self,
                                  current_compliance_rate: float,
                                  target_compliance_rate: float,
                                  annual_violation_cost: float,
                                  implementation_cost: float = 0,
                                  reputation_value: float = 0,
                                  timeframe_years: int = 1) -> Dict[str, Any]:
        """Calculate financial impact of improving regulatory compliance
        
        Args:
            current_compliance_rate: Current compliance rate percentage
            target_compliance_rate: Target compliance rate percentage after improvements
            annual_violation_cost: Annual cost of compliance violations
            implementation_cost: Cost to implement compliance improvements
            reputation_value: Estimated value of reputation improvement
            timeframe_years: Years over which to calculate impact
            
        Returns:
            Dictionary with financial impact results
        """
        try:
            # Calculate violation reduction
            current_violation_rate = 100 - current_compliance_rate
            target_violation_rate = 100 - target_compliance_rate
            violation_reduction_pct = current_violation_rate - target_violation_rate
            
            # Calculate financial savings
            current_violation_cost = annual_violation_cost
            estimated_target_cost = annual_violation_cost * (target_violation_rate / current_violation_rate) if current_violation_rate > 0 else 0
            annual_savings = current_violation_cost - estimated_target_cost
            
            # Add reputation value if provided
            annual_total_benefit = annual_savings + (reputation_value / timeframe_years if reputation_value > 0 else 0)
            total_benefit = annual_total_benefit * timeframe_years
            
            # Calculate ROI
            if implementation_cost > 0:
                roi_pct = ((total_benefit - implementation_cost) / implementation_cost) * 100
                payback_period = implementation_cost / annual_total_benefit if annual_total_benefit > 0 else float('inf')
            else:
                roi_pct = float('inf')
                payback_period = 0
            
            result = {
                "metric": "compliance",
                "current_state": {
                    "compliance_rate": current_compliance_rate,
                    "violation_rate": current_violation_rate,
                    "violation_cost": current_violation_cost
                },
                "target_state": {
                    "compliance_rate": target_compliance_rate,
                    "violation_rate": target_violation_rate,
                    "violation_cost": estimated_target_cost
                },
                "impact": {
                    "compliance_improvement": target_compliance_rate - current_compliance_rate,
                    "violation_reduction_pct": violation_reduction_pct,
                    "annual_cost_savings": annual_savings,
                    "reputation_value": reputation_value,
                    "annual_total_benefit": annual_total_benefit,
                    "total_benefit": total_benefit,
                    "implementation_cost": implementation_cost,
                    "net_benefit": total_benefit - implementation_cost,
                    "roi_percentage": roi_pct,
                    "payback_period_years": payback_period
                },
                "assumptions": {
                    "timeframe_years": timeframe_years,
                    "annual_violation_cost": annual_violation_cost,
                    "reputation_value": reputation_value,
                    "currency": self.currency
                }
            }
            
            self.results["compliance"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error calculating compliance impact: {str(e)}")
            return {"error": f"Compliance calculation failed: {str(e)}"}
    
    def calculate_infrastructure_impact(self,
                                       current_failure_rate: float,
                                       target_failure_rate: float,
                                       average_failure_cost: float,
                                       asset_count: int,
                                       implementation_cost: float = 0,
                                       asset_lifespan_extension_years: float = 0,
                                       asset_replacement_cost: float = 0,
                                       timeframe_years: int = 1) -> Dict[str, Any]:
        """Calculate financial impact of improving infrastructure health
        
        Args:
            current_failure_rate: Current annual failure rate percentage
            target_failure_rate: Target annual failure rate percentage after improvements
            average_failure_cost: Average cost per failure incident
            asset_count: Total number of assets in scope
            implementation_cost: Cost to implement infrastructure improvements
            asset_lifespan_extension_years: Additional years of life for assets (if applicable)
            asset_replacement_cost: Total cost to replace assets at end of life
            timeframe_years: Years over which to calculate impact
            
        Returns:
            Dictionary with financial impact results
        """
        try:
            # Calculate failure reduction
            current_annual_failures = asset_count * (current_failure_rate / 100)
            target_annual_failures = asset_count * (target_failure_rate / 100)
            annual_failure_reduction = current_annual_failures - target_annual_failures
            
            # Calculate direct cost savings from reduced failures
            annual_failure_savings = annual_failure_reduction * average_failure_cost
            
            # Calculate asset life extension value (if applicable)
            extension_value = 0
            if asset_lifespan_extension_years > 0 and asset_replacement_cost > 0:
                # Simplified calculation: Value = (Extension years / Original lifespan) * Replacement cost
                # Assume 30 year original lifespan if not provided
                original_lifespan = self.assumptions.get("original_asset_lifespan", 30)
                extension_value = (asset_lifespan_extension_years / original_lifespan) * asset_replacement_cost
                
                # Amortize over the timeframe for annual value
                annual_extension_value = extension_value / timeframe_years
            else:
                annual_extension_value = 0
            
            # Calculate total annual benefit
            annual_total_benefit = annual_failure_savings + annual_extension_value
            total_benefit = annual_total_benefit * timeframe_years
            
            # Calculate ROI
            if implementation_cost > 0:
                roi_pct = ((total_benefit - implementation_cost) / implementation_cost) * 100
                payback_period = implementation_cost / annual_total_benefit if annual_total_benefit > 0 else float('inf')
            else:
                roi_pct = float('inf')
                payback_period = 0
            
            result = {
                "metric": "infrastructure_health",
                "current_state": {
                    "failure_rate": current_failure_rate,
                    "annual_failures": current_annual_failures,
                    "annual_failure_cost": current_annual_failures * average_failure_cost
                },
                "target_state": {
                    "failure_rate": target_failure_rate,
                    "annual_failures": target_annual_failures,
                    "annual_failure_cost": target_annual_failures * average_failure_cost
                },
                "impact": {
                    "failure_rate_reduction": current_failure_rate - target_failure_rate,
                    "annual_failure_reduction": annual_failure_reduction,
                    "annual_failure_savings": annual_failure_savings,
                    "asset_lifespan_extension": asset_lifespan_extension_years,
                    "asset_extension_value": extension_value,
                    "annual_extension_value": annual_extension_value,
                    "annual_total_benefit": annual_total_benefit,
                    "total_benefit": total_benefit,
                    "implementation_cost": implementation_cost,
                    "net_benefit": total_benefit - implementation_cost,
                    "roi_percentage": roi_pct,
                    "payback_period_years": payback_period
                },
                "assumptions": {
                    "timeframe_years": timeframe_years,
                    "average_failure_cost": average_failure_cost,
                    "asset_count": asset_count,
                    "asset_replacement_cost": asset_replacement_cost,
                    "original_asset_lifespan": self.assumptions.get("original_asset_lifespan", 30),
                    "currency": self.currency
                }
            }
            
            self.results["infrastructure_health"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error calculating infrastructure impact: {str(e)}")
            return {"error": f"Infrastructure calculation failed: {str(e)}"}
    
    def calculate_total_roi(self) -> Dict[str, Any]:
        """Calculate combined ROI across all metrics"""
        if not self.results:
            return {"error": "No financial impact calculations to analyze"}
            
        try:
            # Combine results from all calculations
            total_implementation_cost = 0
            total_annual_benefit = 0
            total_benefit = 0
            
            # Get timeframe from first result (assume same for all)
            timeframe_years = list(self.results.values())[0]["assumptions"]["timeframe_years"]
            
            benefits_by_metric = {}
            costs_by_metric = {}
            
            for metric, result in self.results.items():
                if "impact" in result:
                    impact = result["impact"]
                    total_implementation_cost += impact.get("implementation_cost", 0)
                    
                    # Extract the appropriate annual benefit field based on metric
                    annual_benefit = 0
                    if metric == "water_loss":
                        annual_benefit = impact.get("annual_savings", 0)
                    elif metric == "energy_efficiency":
                        annual_benefit = impact.get("annual_total_savings", 0)
                    elif metric == "compliance":
                        annual_benefit = impact.get("annual_total_benefit", 0)
                    elif metric == "infrastructure_health":
                        annual_benefit = impact.get("annual_total_benefit", 0)
                    else:
                        # Generic fallback
                        for key in ["annual_total_benefit", "annual_total_savings", "annual_savings"]:
                            if key in impact:
                                annual_benefit = impact[key]
                                break
                    
                    total_annual_benefit += annual_benefit
                    
                    # Get the total benefit over the timeframe
                    metric_total_benefit = annual_benefit * timeframe_years
                    total_benefit += metric_total_benefit
                    
                    # Store for breakdown
                    benefits_by_metric[metric] = {
                        "annual": annual_benefit,
                        "total": metric_total_benefit
                    }
                    costs_by_metric[metric] = impact.get("implementation_cost", 0)
            
            # Calculate overall ROI metrics
            if total_implementation_cost > 0:
                overall_roi_pct = ((total_benefit - total_implementation_cost) / total_implementation_cost) * 100
                overall_payback_period = total_implementation_cost / total_annual_benefit if total_annual_benefit > 0 else float('inf')
            else:
                overall_roi_pct = float('inf')
                overall_payback_period = 0
            
            # Prepare summary
            summary = {
                "overall": {
                    "total_implementation_cost": total_implementation_cost,
                    "total_annual_benefit": total_annual_benefit,
                    "total_benefit": total_benefit,
                    "net_benefit": total_benefit - total_implementation_cost,
                    "roi_percentage": overall_roi_pct,
                    "payback_period_years": overall_payback_period
                },
                "breakdown": {
                    "benefits": benefits_by_metric,
                    "costs": costs_by_metric
                },
                "assumptions": {
                    "timeframe_years": timeframe_years,
                    "currency": self.currency,
                    "metrics_included": list(self.results.keys())
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating total ROI: {str(e)}")
            return {"error": f"Total ROI calculation failed: {str(e)}"}

# Main functions to be called from the application
def calculate_roi(inputs: Dict[str, Any], tenant_id: str, currency: str = "£") -> Dict[str, Any]:
    """Calculate ROI for a set of improvements
    
    Args:
        inputs: Dictionary containing metrics and their current/target values
        tenant_id: Tenant identifier
        currency: Currency symbol to use in results
        
    Returns:
        Dictionary with ROI analysis
    """
    calculator = FinancialImpactCalculator(tenant_id, currency)
    
    # Set any global assumptions
    if "assumptions" in inputs:
        calculator.set_assumptions(inputs["assumptions"])
    
    timeframe = inputs.get("timeframe_years", 5)
    
    # Process each metric
    for metric, data in inputs.get("metrics", {}).items():
        if metric == "water_loss" and "current" in data and "target" in data:
            calculator.calculate_water_loss_impact(
                current_loss_pct=data["current"],
                target_loss_pct=data["target"],
                annual_water_volume=data.get("annual_volume", 1000000),
                unit_cost=data.get("unit_cost", 0.5),
                implementation_cost=data.get("implementation_cost", 0),
                timeframe_years=timeframe
            )
        
        elif metric == "energy_efficiency" and "current" in data and "target" in data:
            calculator.calculate_energy_efficiency_impact(
                current_energy_usage=data["current"],
                target_energy_usage=data["target"],
                energy_unit_cost=data.get("unit_cost", 0.15),
                carbon_factor=data.get("carbon_factor", 0.233),
                carbon_price=data.get("carbon_price", 50),
                implementation_cost=data.get("implementation_cost", 0),
                timeframe_years=timeframe
            )
        
        elif metric == "compliance" and "current" in data and "target" in data:
            calculator.calculate_compliance_impact(
                current_compliance_rate=data["current"],
                target_compliance_rate=data["target"],
                annual_violation_cost=data.get("violation_cost", 100000),
                implementation_cost=data.get("implementation_cost", 0),
                reputation_value=data.get("reputation_value", 0),
                timeframe_years=timeframe
            )
        
        elif metric == "infrastructure_health" and "current" in data and "target" in data:
            calculator.calculate_infrastructure_impact(
                current_failure_rate=data["current"],
                target_failure_rate=data["target"],
                average_failure_cost=data.get("failure_cost", 5000),
                asset_count=data.get("asset_count", 100),
                implementation_cost=data.get("implementation_cost", 0),
                asset_lifespan_extension_years=data.get("lifespan_extension", 0),
                asset_replacement_cost=data.get("replacement_cost", 0),
                timeframe_years=timeframe
            )
    
    # Calculate and return overall ROI
    return calculator.calculate_total_roi()

def calculate_financial_impact(metric_type: str, params: Dict[str, Any] = None, tenant_id: str = "default", currency: str = "£") -> Dict[str, Any]:
    """Calculate financial impact for a specific metric
    
    Args:
        metric_type: Type of metric to calculate (water_loss, energy_efficiency, compliance, infrastructure_health)
        params: Parameters specific to the metric type
        tenant_id: Tenant identifier
        currency: Currency symbol to use in results
        
    Returns:
        Dictionary with financial impact analysis
    """
    # Handle case with missing or invalid parameters
    if params is None:
        # Generate reasonable default parameters based on metric type
        if metric_type == "water_loss":
            params = {
                "current_loss": 15,  # Current water loss percentage
                "target_loss": 10,   # Target water loss percentage
                "annual_volume": 10000000,  # Total annual volume in cubic meters
                "unit_cost": 0.5     # Cost per cubic meter
            }
        elif metric_type == "energy_efficiency":
            params = {
                "current_energy": 1200000,  # Current energy usage in kWh
                "target_energy": 1000000,   # Target energy usage in kWh
                "energy_cost": 0.15,        # Cost per kWh
                "carbon_factor": 0.233      # Carbon emission factor kg CO2e per kWh
            }
        elif metric_type == "compliance":
            params = {
                "incident_count": 5,        # Number of compliance incidents
                "average_fine": 50000,     # Average fine amount
                "legal_costs": 25000,      # Legal and admin costs
                "remediation_cost": 100000  # Cost to remediate issues
            }
        elif metric_type == "infrastructure_health":
            params = {
                "maintenance_cost": 500000,  # Current annual maintenance
                "replacement_cost": 2000000, # Cost to replace infrastructure
                "failure_risk": 0.15,       # Probability of failure without investment
                "failure_cost": 1500000     # Cost of failure
            }
    
    calculator = FinancialImpactCalculator(tenant_id, currency)
    
    try:
        if metric_type == "water_loss":
            return calculator.calculate_water_loss_impact(
                current_loss_pct=params.get("current_loss_pct", 15),
                target_loss_pct=params.get("target_loss_pct", 10),
                annual_water_volume=params.get("annual_water_volume", 1000000),
                unit_cost=params.get("unit_cost", 0.5),
                implementation_cost=params.get("implementation_cost", 0),
                timeframe_years=params.get("timeframe_years", 5)
            )
            
        elif metric_type == "energy_efficiency":
            return calculator.calculate_energy_efficiency_impact(
                current_energy_usage=params.get("current_energy_usage", 1000000),
                target_energy_usage=params.get("target_energy_usage", 800000),
                energy_unit_cost=params.get("energy_unit_cost", 0.15),
                carbon_factor=params.get("carbon_factor", 0.233),
                carbon_price=params.get("carbon_price", 50),
                implementation_cost=params.get("implementation_cost", 0),
                timeframe_years=params.get("timeframe_years", 5)
            )
            
        elif metric_type == "compliance":
            return calculator.calculate_compliance_impact(
                current_compliance_rate=params.get("current_compliance_rate", 90),
                target_compliance_rate=params.get("target_compliance_rate", 98),
                annual_violation_cost=params.get("annual_violation_cost", 100000),
                implementation_cost=params.get("implementation_cost", 0),
                reputation_value=params.get("reputation_value", 0),
                timeframe_years=params.get("timeframe_years", 5)
            )
            
        elif metric_type == "infrastructure_health":
            return calculator.calculate_infrastructure_impact(
                current_failure_rate=params.get("current_failure_rate", 5),
                target_failure_rate=params.get("target_failure_rate", 2),
                average_failure_cost=params.get("average_failure_cost", 5000),
                asset_count=params.get("asset_count", 100),
                implementation_cost=params.get("implementation_cost", 0),
                asset_lifespan_extension_years=params.get("asset_lifespan_extension_years", 0),
                asset_replacement_cost=params.get("asset_replacement_cost", 0),
                timeframe_years=params.get("timeframe_years", 5)
            )
            
        else:
            return {"error": f"Unknown metric type: {metric_type}"}
            
    except Exception as e:
        logger.error(f"Error in financial impact calculation: {str(e)}")
        return {"error": f"Calculation failed: {str(e)}"}

# For testing
if __name__ == "__main__":
    # Test individual calculation
    water_impact = calculate_financial_impact(
        "water_loss",
        {
            "current_loss_pct": 15,
            "target_loss_pct": 10,
            "annual_water_volume": 5000000,  # 5 million cubic meters
            "unit_cost": 0.6,  # £0.60 per cubic meter
            "implementation_cost": 500000,  # £500,000 implementation cost
            "timeframe_years": 5
        },
        "test_tenant"
    )
    
    # Test combined ROI calculation
    combined_roi = calculate_roi(
        {
            "timeframe_years": 5,
            "metrics": {
                "water_loss": {
                    "current": 15,
                    "target": 10,
                    "annual_volume": 5000000,
                    "unit_cost": 0.6,
                    "implementation_cost": 500000
                },
                "energy_efficiency": {
                    "current": 1000000,  # 1 million kWh
                    "target": 800000,    # 800,000 kWh
                    "unit_cost": 0.15,   # £0.15 per kWh
                    "implementation_cost": 300000
                },
                "compliance": {
                    "current": 90,  # 90% compliance
                    "target": 98,   # 98% compliance
                    "violation_cost": 200000,  # £200,000 annual violation costs
                    "implementation_cost": 150000
                }
            }
        },
        "test_tenant"
    )
    
    # Print results
    print("Water Loss Impact:")
    print(json.dumps(water_impact, indent=2))
    print("\nCombined ROI:")
    print(json.dumps(combined_roi, indent=2))
