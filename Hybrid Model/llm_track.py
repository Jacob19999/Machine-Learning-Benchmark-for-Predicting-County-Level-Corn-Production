"""
LLM Track - Bounded Reasoning with Grok API
Calls Grok API to get adjustment factor a ∈ [-0.20, 0.20] with strict JSON response
Uses PDF fact sheets for county information
"""

import json
import requests
import numpy as np
from typing import Dict, Optional, Tuple
import time
from pathlib import Path
import PyPDF2


class LLMTrack:
    """
    LLM Track for bounded reasoning
    Returns adjustment a ∈ [-0.20, 0.20] with short driver bullets
    Uses PDF fact sheets for county information
    """
    
    def __init__(self, api_key: str, fact_sheet_dir: str = "Fact Sheet", 
                 model: str = "grok-4-latest", temperature: float = 0.0):
        """
        Initialize LLM Track
        
        Args:
            api_key: Grok API key
            fact_sheet_dir: Directory containing PDF fact sheets (default: "Fact Sheet")
            model: Model name (default: grok-4-latest)
            temperature: Temperature for generation (0.0 for deterministic)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Set up fact sheet directory
        self.fact_sheet_dir = Path(__file__).parent / fact_sheet_dir
        if not self.fact_sheet_dir.exists():
            raise ValueError(f"Fact sheet directory not found: {self.fact_sheet_dir}")
    
    def _read_pdf(self, fips: int) -> str:
        """
        Read PDF fact sheet for given FIPS code
        
        Args:
            fips: County FIPS code
            
        Returns:
            Extracted text from PDF
        """
        pdf_path = self.fact_sheet_dir / f"{fips}.pdf"
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fact sheet not found for FIPS {fips}: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise ValueError(f"Error reading PDF for FIPS {fips}: {e}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with conservative adjustment rules"""
        return """You are a conservative adjustment module for county-level corn yield.

Your ONLY inputs are a structured "factsheet" with exogenous signals that the ML model does not capture well.

You must ignore agronomic variables already modeled by ML (soils, ethanol distance, historical yields, GLDAS/PRISM windowed indices, USDM, NDVI, etc.).

Goal: Propose a small, bounded multiplicative adjustment to the provided ML baseline prediction based solely on the allowed facts.

Hard constraints

Use only these fields if present:

economy_population, economy_unemployment_rate_pct, economy_available_workers,

economy_required_hourly_wage_single_usd, economy_median_hourly_wage_region_usd,

economy_median_household_income_usd, selected industry wage/size metrics.

market_basis_anomaly_usd, elevator_outage_flag, rail_disruption_flag.

prevented_planting_flag, pp_share_pct, insurance_disaster_flag.

hail_event_flag, derecho_event_flag, localized_flood_flag.

nass_planting_progress_pct, nass_harvest_progress_pct (weekly progress summaries).

ML meta: ml_pred (bu/ac), ml_band_width (U−L), train_rmse.

Forbidden: any soils, ethanol distance, recent yields, or windowed agronomic/drought/canopy variables. If such fields appear, ignore them.

Compute a raw signed adjustment adj_raw in [-1, +1] reflecting only exogenous stress/support.

Negative signals (examples): very high unemployment, severe labor shortages, high required_hourly_wage vs regional median, active logistics disruptions, hail/derecho flags, widespread prevented planting.

Positive signals (examples): strong labor availability with low wage pressure and no disruptions.

The final applied adjustment is bounded and damped by ML confidence:

alpha = 0.15 (max magnitude).

kappa = exp( - ml_band_width / train_rmse ).

adj = clip(adj_raw, -alpha, +alpha) * (1 - kappa).

Calibrated caution: If evidence is mixed or weak, set adj_raw = 0.0. Never exceed the bounds.

Output JSON only with this schema (numbers only, no explanations in prose):

For a SINGLE county:
{
  "adj_raw": float,                  // in [-1, 1], before damping
  "adj": float,                      // after bounds & damping rule above
  "pred_yield_bu_ac": float,         // = ml_pred * (1 + adj)
  "low_80": float,                   // = (ml_pred - 0.5*ml_band_width) * (1 + 0.7*adj)
  "high_80": float,                  // = (ml_pred + 0.5*ml_band_width) * (1 + 0.7*adj)
  "drivers": [ "short phrase", ... ] // 3–5 concise drivers referencing ONLY allowed fields
}

For MULTIPLE counties (differentiated by FIPS):
{
  "<fips_code>": {
    "adj_raw": float,
    "adj": float,
    "pred_yield_bu_ac": float,
    "low_80": float,
    "high_80": float,
    "drivers": [ "short phrase", ... ]
  },
  "<fips_code>": { ... },
  ...
}

No chain-of-thought. Do not explain your reasoning; just fill the JSON.

If required fields are missing (e.g., ml_pred, ml_band_width, train_rmse), return:
{"adj_raw": 0.0, "adj": 0.0, "pred_yield_bu_ac": null, "low_80": null, "high_80": null, "drivers": ["insufficient_fields"]}

FIPS codes should be strings (e.g., "27129" not 27129)."""
    
    def _build_user_prompt(self, factsheet_text: str, ml_baseline: float) -> str:
        """Build user prompt with factsheet and ML baseline"""
        return f"""Analyze this corn production factsheet and provide a bounded adjustment to the ML baseline prediction.

ML Baseline Prediction: {ml_baseline:,.0f} bushels

Fact Sheet Information:
{factsheet_text}

Provide your analysis as JSON with:
- "adjustment": float in [-0.20, 0.20] representing multiplicative adjustment factor
- "drivers": array of 2-4 short bullet points explaining key factors

Example format:
{{
  "adjustment": -0.05,
  "drivers": [
    "Severe moisture deficit during critical growth period",
    "High VPD indicates elevated water stress",
    "Recent yields declining trend suggests production challenges"
  ]
}}

Return ONLY the JSON object, no other text."""
    
    def _call_api(self, prompt: str) -> Optional[Dict]:
        """
        Call Grok API with retry logic
        
        Args:
            prompt: User prompt text
            
        Returns:
            API response or None if failed
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "stream": False,
            "temperature": self.temperature
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                print(f"API call failed after {self.max_retries} attempts: {e}")
                return None
    
    def _parse_response(self, api_response: Dict) -> Tuple[Optional[float], Optional[list]]:
        """
        Parse API response to extract adjustment and drivers
        
        Args:
            api_response: API response dictionary
            
        Returns:
            Tuple of (adjustment, drivers) or (None, None) if parsing fails
        """
        try:
            content = api_response['choices'][0]['message']['content'].strip()
            
            # Try to extract JSON from response (handle markdown code blocks)
            if content.startswith('```'):
                # Extract JSON from code block
                lines = content.split('\n')
                json_start = None
                json_end = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('```'):
                        if json_start is None:
                            json_start = i + 1
                        else:
                            json_end = i
                            break
                if json_start and json_end:
                    content = '\n'.join(lines[json_start:json_end])
                elif json_start:
                    content = '\n'.join(lines[json_start:])
            
            # Parse JSON
            result = json.loads(content)
            
            adjustment = float(result.get('adjustment', 0.0))
            drivers = result.get('drivers', [])
            
            # Validate adjustment is in bounds
            if not (-0.20 <= adjustment <= 0.20):
                print(f"Warning: Adjustment {adjustment} out of bounds, clipping to [-0.20, 0.20]")
                adjustment = np.clip(adjustment, -0.20, 0.20)
            
            return adjustment, drivers
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Failed to parse API response: {e}")
            print(f"Response content: {content[:500] if 'content' in locals() else 'N/A'}")
            return None, None
    
    def get_adjustment(self, fips: int, ml_baseline: float) -> Tuple[Optional[float], Optional[list], bool]:
        """
        Get adjustment factor from LLM using PDF fact sheet
        
        Args:
            fips: County FIPS code
            ml_baseline: ML baseline prediction (for context in prompt)
            
        Returns:
            Tuple of (adjustment, drivers, success)
            - adjustment: float in [-0.20, 0.20] or None if failed
            - drivers: list of driver bullets or None if failed
            - success: bool indicating if parsing succeeded
        """
        # Read PDF fact sheet
        try:
            factsheet_text = self._read_pdf(fips)
        except Exception as e:
            print(f"Error reading fact sheet for FIPS {fips}: {e}")
            return None, None, False
        
        # Build prompt with factsheet and ML baseline
        prompt = self._build_user_prompt(factsheet_text, ml_baseline)
        
        # Call API
        api_response = self._call_api(prompt)
        
        if api_response is None:
            return None, None, False
        
        adjustment, drivers = self._parse_response(api_response)
        
        if adjustment is None:
            return None, None, False
        
        return adjustment, drivers, True
    
    def get_adjustment_with_fallback(self, fips: int, ml_baseline: float,
                                     fallback_adjustment: float = 0.0) -> Tuple[float, Optional[list]]:
        """
        Get adjustment with fallback to default if parsing fails
        
        Args:
            fips: County FIPS code
            ml_baseline: ML baseline prediction
            fallback_adjustment: Default adjustment if API/parsing fails
            
        Returns:
            Tuple of (adjustment, drivers)
        """
        adjustment, drivers, success = self.get_adjustment(fips, ml_baseline)
        
        if not success:
            print(f"LLM parsing failed for FIPS {fips}, using fallback adjustment: {fallback_adjustment}")
            return fallback_adjustment, None
        
        return adjustment, drivers

