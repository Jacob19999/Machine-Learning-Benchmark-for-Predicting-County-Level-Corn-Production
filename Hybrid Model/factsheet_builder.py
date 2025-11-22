"""
Factsheet Builder for LLM Track
Constructs numeric factsheet per (county, year) for LLM reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class FactsheetBuilder:
    """
    Builds numeric factsheet per (county, year) containing:
    - Soils (AWC, OM, drainage) - approximated from available data
    - CDL fraction - placeholder (would need CDL data)
    - Ethanol distance
    - Pre-season price signals (Dec futures avg/vol, basis, prior-harvest cash) - placeholder
    - Windowed indices: MB_w, VPD, thermal, radiation, root moisture, wind
    - Drought %D2+_w - placeholder (would need drought monitor data)
    - Recent yields y_c,t-1, y_c,t-2, y_c,t-3
    - ML baseline y^_ML
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with consolidated dataset
        
        Args:
            data: DataFrame with columns including fips, year, county_name, 
                  and all environmental/economic features
        """
        self.data = data.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for factsheet construction"""
        # Ensure year and fips are present
        if 'year' not in self.data.columns:
            raise ValueError("Data must contain 'year' column")
        if 'fips' not in self.data.columns:
            raise ValueError("Data must contain 'fips' column")
        
        # Sort by county and year for lag calculations
        self.data = self.data.sort_values(['fips', 'year', 'month'] if 'month' in self.data.columns else ['fips', 'year'])
    
    def get_recent_yields(self, fips: int, year: int) -> Dict[str, float]:
        """
        Get recent yields for county: y_c,t-1, y_c,t-2, y_c,t-3
        
        Args:
            fips: County FIPS code
            year: Target year
            
        Returns:
            Dict with y_t_minus_1, y_t_minus_2, y_t_minus_3 (or NaN if not available)
        """
        county_data = self.data[self.data['fips'] == fips].copy()
        
        # Aggregate to yearly if monthly data
        if 'month' in county_data.columns:
            county_data = county_data.groupby('year').agg({
                'corn_production_bu': 'sum' if 'corn_production_bu' in county_data.columns else 'first'
            }).reset_index()
        
        yields = {}
        for lag in [1, 2, 3]:
            target_year = year - lag
            year_data = county_data[county_data['year'] == target_year]
            if len(year_data) > 0 and 'corn_production_bu' in year_data.columns:
                yields[f'y_t_minus_{lag}'] = float(year_data['corn_production_bu'].iloc[0])
            else:
                yields[f'y_t_minus_{lag}'] = np.nan
        
        return yields
    
    def get_windowed_indices(self, fips: int, year: int, window_months: list = None) -> Dict[str, float]:
        """
        Get windowed indices for growing season
        
        Args:
            fips: County FIPS code
            year: Target year
            window_months: List of months to include (default: Apr-Sep for growing season)
            
        Returns:
            Dict with MB_w, VPD, thermal, radiation, root_moisture, wind
        """
        if window_months is None:
            window_months = [4, 5, 6, 7, 8, 9]  # Apr-Sep growing season
        
        county_data = self.data[
            (self.data['fips'] == fips) & 
            (self.data['year'] == year)
        ].copy()
        
        if 'month' in county_data.columns:
            county_data = county_data[county_data['month'].isin(window_months)]
        
        if len(county_data) == 0:
            return {
                'MB_w': np.nan,
                'VPD_w': np.nan,
                'thermal_w': np.nan,
                'radiation_w': np.nan,
                'root_moisture_w': np.nan,
                'wind_w': np.nan
            }
        
        indices = {}
        
        # MB_w: Moisture Balance (precipitation - evapotranspiration)
        if 'prism_ppt_in' in county_data.columns and 'Evap_tavg' in county_data.columns:
            ppt = county_data['prism_ppt_in'].sum() if 'month' in county_data.columns else county_data['prism_ppt_in'].iloc[0]
            evap = county_data['Evap_tavg'].mean() if 'month' in county_data.columns else county_data['Evap_tavg'].iloc[0]
            indices['MB_w'] = ppt - evap
        else:
            indices['MB_w'] = np.nan
        
        # VPD_w: Vapor Pressure Deficit (use PRISM VPD if available)
        if 'prism_vpdmax_hpa' in county_data.columns:
            indices['VPD_w'] = county_data['prism_vpdmax_hpa'].mean() if 'month' in county_data.columns else county_data['prism_vpdmax_hpa'].iloc[0]
        elif 'prism_vpdmin_hpa' in county_data.columns:
            indices['VPD_w'] = county_data['prism_vpdmin_hpa'].mean() if 'month' in county_data.columns else county_data['prism_vpdmin_hpa'].iloc[0]
        else:
            indices['VPD_w'] = np.nan
        
        # Thermal: Average temperature
        if 'prism_tmean_degf' in county_data.columns:
            indices['thermal_w'] = county_data['prism_tmean_degf'].mean() if 'month' in county_data.columns else county_data['prism_tmean_degf'].iloc[0]
        elif 'Tair_f_inst' in county_data.columns:
            # Convert Kelvin to Fahrenheit
            temp_k = county_data['Tair_f_inst'].mean() if 'month' in county_data.columns else county_data['Tair_f_inst'].iloc[0]
            indices['thermal_w'] = (temp_k - 273.15) * 9/5 + 32
        else:
            indices['thermal_w'] = np.nan
        
        # Radiation: Shortwave downwelling
        if 'SWdown_f_tavg' in county_data.columns:
            indices['radiation_w'] = county_data['SWdown_f_tavg'].mean() if 'month' in county_data.columns else county_data['SWdown_f_tavg'].iloc[0]
        else:
            indices['radiation_w'] = np.nan
        
        # Root moisture: RootMoist_inst
        if 'RootMoist_inst' in county_data.columns:
            indices['root_moisture_w'] = county_data['RootMoist_inst'].mean() if 'month' in county_data.columns else county_data['RootMoist_inst'].iloc[0]
        else:
            indices['root_moisture_w'] = np.nan
        
        # Wind: Wind_f_inst
        if 'Wind_f_inst' in county_data.columns:
            indices['wind_w'] = county_data['Wind_f_inst'].mean() if 'month' in county_data.columns else county_data['Wind_f_inst'].iloc[0]
        else:
            indices['wind_w'] = np.nan
        
        return indices
    
    def get_soil_properties(self, fips: int, year: int) -> Dict[str, float]:
        """
        Get soil properties (AWC, OM, drainage)
        Note: These are approximated from available soil moisture data
        In production, would use actual SSURGO/STATSGO data
        
        Args:
            fips: County FIPS code
            year: Target year
            
        Returns:
            Dict with AWC (approximated), OM (placeholder), drainage (placeholder)
        """
        county_data = self.data[
            (self.data['fips'] == fips) & 
            (self.data['year'] == year)
        ].copy()
        
        if len(county_data) == 0:
            return {'AWC': np.nan, 'OM': np.nan, 'drainage': np.nan}
        
        # Approximate AWC from soil moisture capacity
        # Use average of soil moisture across depths as proxy
        soil_moisture_cols = [col for col in county_data.columns if 'SoilMoi' in col]
        if soil_moisture_cols:
            if 'month' in county_data.columns:
                awc_proxy = county_data[soil_moisture_cols].mean().mean()
            else:
                awc_proxy = county_data[soil_moisture_cols].iloc[0].mean()
        else:
            awc_proxy = np.nan
        
        return {
            'AWC': float(awc_proxy) if not np.isnan(awc_proxy) else np.nan,
            'OM': np.nan,  # Would need organic matter data
            'drainage': np.nan  # Would need drainage class data
        }
    
    def get_price_signals(self, fips: int, year: int) -> Dict[str, float]:
        """
        Get pre-season price signals
        Note: Dec futures, basis, prior-harvest cash are placeholders
        Would need actual commodity price data
        
        Args:
            fips: County FIPS code
            year: Target year
            
        Returns:
            Dict with dec_futures_avg, dec_futures_vol, basis, prior_harvest_cash
        """
        # Placeholder - would need actual futures/price data
        return {
            'dec_futures_avg': np.nan,
            'dec_futures_vol': np.nan,
            'basis': np.nan,
            'prior_harvest_cash': np.nan
        }
    
    def get_drought_index(self, fips: int, year: int) -> Dict[str, float]:
        """
        Get drought %D2+_w (percent of county in D2+ drought)
        Note: Would need US Drought Monitor data
        
        Args:
            fips: County FIPS code
            year: Target year
            
        Returns:
            Dict with drought_pct_D2plus
        """
        return {
            'drought_pct_D2plus': np.nan  # Would need drought monitor data
        }
    
    def build_factsheet(self, fips: int, year: int, ml_baseline: float, 
                       cdl_fraction: Optional[float] = None) -> Dict[str, float]:
        """
        Build complete numeric factsheet for (county, year)
        
        Args:
            fips: County FIPS code
            year: Target year
            ml_baseline: ML model prediction y^_ML
            cdl_fraction: CDL fraction (optional, placeholder)
            
        Returns:
            Complete factsheet dictionary
        """
        factsheet = {
            'fips': int(fips),
            'year': int(year),
            'ml_baseline': float(ml_baseline)
        }
        
        # Get county name if available
        county_data = self.data[self.data['fips'] == fips]
        if len(county_data) > 0 and 'county_name' in county_data.columns:
            factsheet['county_name'] = county_data['county_name'].iloc[0]
        
        # Soils
        soil_props = self.get_soil_properties(fips, year)
        factsheet.update(soil_props)
        
        # CDL fraction
        factsheet['cdl_fraction'] = float(cdl_fraction) if cdl_fraction is not None else np.nan
        
        # Ethanol distance
        county_data = self.data[
            (self.data['fips'] == fips) & 
            (self.data['year'] == year)
        ]
        if len(county_data) > 0 and 'dist_km_ethanol' in county_data.columns:
            factsheet['ethanol_distance'] = float(county_data['dist_km_ethanol'].iloc[0])
        else:
            factsheet['ethanol_distance'] = np.nan
        
        # Pre-season price signals
        price_signals = self.get_price_signals(fips, year)
        factsheet.update(price_signals)
        
        # Windowed indices
        windowed = self.get_windowed_indices(fips, year)
        factsheet.update(windowed)
        
        # Drought
        drought = self.get_drought_index(fips, year)
        factsheet.update(drought)
        
        # Recent yields
        recent_yields = self.get_recent_yields(fips, year)
        factsheet.update(recent_yields)
        
        return factsheet
    
    def format_factsheet_for_llm(self, factsheet: Dict[str, float]) -> str:
        """
        Format factsheet as text prompt for LLM
        
        Args:
            factsheet: Factshet dictionary
            
        Returns:
            Formatted string for LLM prompt
        """
        lines = [
            f"County: {factsheet.get('county_name', 'Unknown')} (FIPS: {factsheet['fips']}, Year: {factsheet['year']})",
            "",
            "SOIL PROPERTIES:",
            f"  AWC (Available Water Capacity): {factsheet.get('AWC', 'N/A'):.2f}",
            f"  OM (Organic Matter): {factsheet.get('OM', 'N/A')}",
            f"  Drainage: {factsheet.get('drainage', 'N/A')}",
            "",
            "AGRICULTURAL CONTEXT:",
            f"  CDL Fraction: {factsheet.get('cdl_fraction', 'N/A')}",
            f"  Ethanol Distance (km): {factsheet.get('ethanol_distance', 'N/A'):.2f}",
            "",
            "PRE-SEASON PRICE SIGNALS:",
            f"  Dec Futures Avg: {factsheet.get('dec_futures_avg', 'N/A')}",
            f"  Dec Futures Vol: {factsheet.get('dec_futures_vol', 'N/A')}",
            f"  Basis: {factsheet.get('basis', 'N/A')}",
            f"  Prior Harvest Cash: {factsheet.get('prior_harvest_cash', 'N/A')}",
            "",
            "WINDOWED INDICES (Growing Season):",
            f"  MB_w (Moisture Balance): {factsheet.get('MB_w', 'N/A'):.2f}",
            f"  VPD_w (Vapor Pressure Deficit): {factsheet.get('VPD_w', 'N/A'):.2f}",
            f"  Thermal_w (Temperature): {factsheet.get('thermal_w', 'N/A'):.2f} °F",
            f"  Radiation_w: {factsheet.get('radiation_w', 'N/A'):.2f} W/m²",
            f"  Root Moisture_w: {factsheet.get('root_moisture_w', 'N/A'):.2f}",
            f"  Wind_w: {factsheet.get('wind_w', 'N/A'):.2f} m/s",
            "",
            "DROUGHT:",
            f"  %D2+_w: {factsheet.get('drought_pct_D2plus', 'N/A')}%",
            "",
            "RECENT YIELDS:",
            f"  y_t-1: {factsheet.get('y_t_minus_1', 'N/A'):,.0f} bushels" if not np.isnan(factsheet.get('y_t_minus_1', np.nan)) else "  y_t-1: N/A",
            f"  y_t-2: {factsheet.get('y_t_minus_2', 'N/A'):,.0f} bushels" if not np.isnan(factsheet.get('y_t_minus_2', np.nan)) else "  y_t-2: N/A",
            f"  y_t-3: {factsheet.get('y_t_minus_3', 'N/A'):,.0f} bushels" if not np.isnan(factsheet.get('y_t_minus_3', np.nan)) else "  y_t-3: N/A",
            "",
            f"ML BASELINE PREDICTION: {factsheet['ml_baseline']:,.0f} bushels"
        ]
        
        return "\n".join(lines)

