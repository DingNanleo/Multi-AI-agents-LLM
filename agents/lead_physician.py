from agents.base_agent import BaseAgent
from typing import List, Dict
import json
import re

class LeadPhysician(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Johnson", "Lead Physician")
        
    def perform_task(self, specialist_opinions: List[Dict]) -> Dict:
        """Analyzes opinions from specialists and returns integrated analysis"""
        # Create structured summary of all opinions
        opinions_summary = "\n\n".join(
            f"### {op['Agent_Name']} Opinion:\n"
            f"Reasoning: {op['Reasoning']}\n"
            f"Choice: {self._extract_choice(op['Choice'])}\n"
            for op in specialist_opinions
        )
    
        prompt = f"""As {self.name} ({self.role}), analyze specialist opinions:

{opinions_summary}

Systematically classify the information into these categories:

1. Consistency: Identify diagnoses and treatment recommendations that are similar or identical across specialists.
2. Conflict: Highlight contradictory diagnoses or treatment choices among specialists.
3. Independence: Extract unique insights mentioned by individual specialists not addressed by others.
4. Integration: Synthesize a structured summary incorporating all perspectives.

Return STRICTLY in this JSON format (less than 3 sentences):
{{
    "consistency": ["List all points of agreement between specialists"],
    "conflict": ["List any conflicting points or empty if none"],
    "independence": {{
        "Ob/gyn Specialist": ["Unique points from OB/GYN"],
        "Urologist Specialist": ["Unique points from Urologist"]
    }},
    "integration": ["Comprehensive synthesis of all opinions with clinical recommendation"]
}}"""
        
        #print("\n====Lead physician Prompt Input====")
        #print(prompt)

        raw_response = self.call_llm(prompt, max_tokens=800)
        
        #print("\n====Lead physician (Raw) Output====")
        #print(raw_response)

        return self._parse_response(raw_response)
    
    def _extract_choice(self, choice_text: str) -> str:
        """Extracts clean choice from Choice format"""
        match = re.search(r'Choice:\s*(\{.*?\}:\s*\{.*?\})', choice_text)
        return match.group(1) if match else choice_text
    
    def _parse_response(self, response: str) -> Dict:
        """Parses LLM response into structured data with 3 fallback methods"""
        # Method 1: Try direct JSON parsing
        try:
            clean_response = response.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_response)
        except json.JSONDecodeError:
            pass
            
        # Method 3: Return error if all parsing fails
        return {
            "error": "Failed to parse LLM response",
            "raw_response": response
        }