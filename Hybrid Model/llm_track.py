"""
LLM Track - Bounded Reasoning with Grok API
Calls Grok API to get adjustment factor a ∈ [-0.20, 0.20] with strict JSON response
"""

import json
import requests
import numpy as np
from typing import Dict, Optional, Tuple
import time


class LLMTrack:
    """
    LLM Track for bounded reasoning
    Returns adjustment a ∈ [-0.20, 0.20] with short driver bullets
    """
    
    def __init__(self, api_key: str, model: str = "grok-4-latest", temperature: float = 0.0):
        """
        Initialize LLM Track
        
        Args:
            api_key: Grok API key
            model: Model name (default: grok-beta)
            temperature: Temperature for generation (0.0 for deterministic)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with agronomic rules"""
        return """You are an agricultural yield prediction expert. Your task is to analyze county-level corn production factsheets and provide a bounded adjustment factor to refine ML model predictions.

RULES:
1. You must return STRICT JSON only, no free-form text or chain-of-thought
2. The adjustment factor 'a' must be in the range [-0.20, 0.20]
3. Include 2-4 short driver bullets explaining key factors
4. Format: {"adjustment": <float>, "drivers": ["bullet1", "bullet2", ...]}

AGRONOMIC RULES (embedded):
- If Reproductive MB_w << 0 (severe moisture deficit) AND VPD high → negative adjustment
- If AWC low AND VPD high → stronger negative adjustment
- If Root Moisture low AND Thermal high → negative adjustment
- If Recent yields declining (y_t-1 < y_t-2 < y_t-3) → negative adjustment
- If Windowed indices favorable (MB_w > 0, VPD moderate, Thermal optimal) → positive adjustment
- If ML baseline seems high relative to recent yields → negative adjustment
- If Drought %D2+ high → negative adjustment
- If Pre-season prices strong → slight positive adjustment (economic signal)

Return ONLY valid JSON, no other text."""
    
    def _build_user_prompt(self, factsheet_text: str) -> str:
        """Build user prompt with factsheet"""
        return f"""Analyze this corn production factsheet and provide a bounded adjustment to the ML baseline prediction.

{factsheet_text}

Provide your analysis as JSON with:
- "adjustment": float in [-0.20, 0.20] representing multiplicative adjustment factor
- "drivers": array of 2-4 short bullet points explaining key factors

Example format:
{{
  "adjustment": -0.05,
  "drivers": [
    "Severe moisture deficit (MB_w = -2.3) during critical growth period",
    "High VPD (8.5) indicates elevated water stress",
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
    
    def get_adjustment(self, factsheet_text: str) -> Tuple[Optional[float], Optional[list], bool]:
        """
        Get adjustment factor from LLM
        
        Args:
            factsheet_text: Formatted factsheet text
            
        Returns:
            Tuple of (adjustment, drivers, success)
            - adjustment: float in [-0.20, 0.20] or None if failed
            - drivers: list of driver bullets or None if failed
            - success: bool indicating if parsing succeeded
        """
        api_response = self._call_api(factsheet_text)
        
        if api_response is None:
            return None, None, False
        
        adjustment, drivers = self._parse_response(api_response)
        
        if adjustment is None:
            return None, None, False
        
        return adjustment, drivers, True
    
    def get_adjustment_with_fallback(self, factsheet_text: str, 
                                     fallback_adjustment: float = 0.0) -> Tuple[float, Optional[list]]:
        """
        Get adjustment with fallback to default if parsing fails
        
        Args:
            factsheet_text: Formatted factsheet text
            fallback_adjustment: Default adjustment if API/parsing fails
            
        Returns:
            Tuple of (adjustment, drivers)
        """
        adjustment, drivers, success = self.get_adjustment(factsheet_text)
        
        if not success:
            print(f"LLM parsing failed, using fallback adjustment: {fallback_adjustment}")
            return fallback_adjustment, None
        
        return adjustment, drivers

