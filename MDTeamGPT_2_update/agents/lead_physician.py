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

1. Consistency: Aggregate similar or identical recommendations.
2. Conflict: Identify and document differing opinions among specialists.
3. Independence: Extract insights that were mentioned by only one specialist.
4. Integration: Formulate a structured summary incorporating all perspectives.

Return STRICTLY in this JSON format (less than 3 sentences):
{{
    "consistency": ["Summarize the common aspects found across multiple agents."],
    "conflict": ["List conflicting points between agents; leave empty if no conflicts exist."],
    "independence": ["Extract unique viewpoints mentioned by only one agent; leave empty if none."],
    "integration": ["rovide a well-structured summary integrating all perspectives."]
}}"""  
        print("\n====Lead physician Prompt Input====")
        print(prompt)

        raw_response = self.call_llm(prompt, max_tokens=800)
        print("\n====Lead physician raw Output====")
        print(raw_response)

        parsed_response=self._parse_response(raw_response)
        print("\n====Lead physician parsed Output====")
        print(parsed_response)

        return parsed_response
    
    def _extract_choice(self, choice_text: str) -> str:
        """Extracts clean choice from Choice format"""
        match = re.search(r'Choice:\s*(\{.*?\}:\s*\{.*?\})', choice_text)
        return match.group(1) if match else choice_text
    
    def _parse_response(self, response: str) -> Dict:
        """Parses LLM response into structured data with multiple fallback methods"""
        # Method 1: Try extracting JSON from markdown code blocks
        try:
            code_blocks = re.findall(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
            if code_blocks:
                return json.loads(code_blocks[0].strip())
        except json.JSONDecodeError:
            pass
        
        # Method 2: Try finding JSON in the response (loose parsing)
        try:
            # Look for the first { and last } in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Method 3: Return error if all parsing fails
        return {
            "error": "Failed to parse LLM response",
            "raw_response": response
        }


    # def _parse_response(self, response: str) -> Dict:
    #     """Parses LLM response into structured data with 3 fallback methods"""
    #     # Method 1: Try direct JSON parsing
    #     try:
    #         clean_response = response.replace('```json', '').replace('```', '').strip()
    #         return json.loads(clean_response)
    #     except json.JSONDecodeError:
    #         pass
            
    #     # Method 3: Return error if all parsing fails
    #     return {
    #         "error": "Failed to parse LLM response",
    #         "raw_response": response
    #     }