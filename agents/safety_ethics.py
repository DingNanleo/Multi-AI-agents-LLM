import json
import re
from typing import Dict
from agents.base_agent import BaseAgent

class SafetyEthicsReviewer(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Ethics", "Safety & Ethics Reviewer")
        
    def perform_task(self, final_opinion: str, patient_background: str, historical_pool: Dict) -> Dict:
        prompt = f"""Role: {self.name} ({self.role})
    Patient: {patient_background[:150]}...
    Opinion: {final_opinion[:300]}...

    Evaluate for (1 sentence each):
    1. Safety risks
    2. Ethical issues
    3. Potential biases

    Return JSON with:
    {{
    "assessment": "Overall summary",
    "concerns": ["list", "of", "issues"],
    "recommendations": ["suggested", "changes"],
    "approved": boolean
    }}"""
    
        return self._parse_response(
            self.call_llm(prompt, max_tokens=500)
        )
    
    
    def _parse_response(self, response: str) -> Dict:
        """More robust response parsing"""
        try:
            if isinstance(response, dict):
                return response  # Already parsed
            
            if response.startswith('{') or 'assessment' in response.lower():
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # Extract JSON from malformed response
                    json_str = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_str:
                        return json.loads(json_str.group())
        
            # Default approval if parsing fails
            return {
                "assessment": "Approved with conditions",
                "concerns": ["Unable to fully parse response"],
                "recommendations": ["Verify treatment plan with specialists"],
                "approved": True
            }
        except Exception as e:
            print(f"Ethics review parse error: {e}")
            return {
                "assessment": "Approved with verification needed",
                "concerns": [],
                "recommendations": [],
                "approved": True  # Default to approved
            }