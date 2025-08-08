from agents.base_agent import BaseAgent
from typing import Dict, List, Optional, Union, Tuple
import json
import re

###### primary_care0.py featue: just specialists ########

class PrimaryCareDoctor(BaseAgent):
    def __init__(self, max_retries: int = 10):
        super().__init__("Dr. Smith", "Primary Care Physician")
        self.max_retries = max_retries  # Maximum retry attempts

    def perform_task(self, question: str, options: str) -> Dict:
        print("\n====== Primary Care Prompt ======")
        prompt = self._build_prompt(question, options)
        print(prompt)

        print(f"\n====== Primary Care LLM Response======")
        llm_response = self.call_llm(prompt, require_json=True)
        print(llm_response)

        parsed_response, is_valid = self._parse_and_validate_response(llm_response)
        
        retry_count = 0
        while not is_valid and retry_count < self.max_retries:
            feedback = self._generate_feedback(parsed_response) if parsed_response else "Invalid response format"
            corrected_response = self._retry_with_feedback(prompt, feedback)
            print(f"\n====== Primary Care LLM Response======")
            print(corrected_response)
            parsed_response, is_valid = self._parse_and_validate_response(corrected_response)
            retry_count += 1
        
        if not is_valid:
            return {
                "primary_care_choices": []
            }
        
        print("\n====== Primary Care Parsed Response ======")
        print(parsed_response)
        return parsed_response

    def _build_prompt(self, question: str, options: List[str]) -> str:
        return f"""
        [ROLE]
        You are an expert primary care physician. Analyze the clinical scenario and recommend:
        1. Exactly 3-5 relevant medical specialists
        2. A brief description of their role
        3. Your clinical reasoning

        [CASE DETAILS]
        Patient Presentation and Medical Question: {question}

        [REQUIRED OUTPUT FORMAT]
        [
        {{
            "specialist": "Standard medical specialty name (e.g., Cardiologist)",
            "specialist_role_description": "1-2 sentence description",
            "rationale": "Why this specialist is appropriate based on symptoms"
        }},
        {{
            "specialist": "Another standard specialty",
            "specialist_role_description": "Description",
            "rationale": "Reasoning"
        }}
        ]
        
        [STRICT RULES]
        - You MUST return exactly 3-5 specialists in an array.
        - Each entry MUST have EXACTLY these 3 keys: "specialist", "specialist_role_description", "rationale".
        - Keys MUST be in double quotes.
        - Specialist names MUST be standard medical specialties.
        - If uncertain about any field, use "Unknown" as the value.
        - Do NOT include any additional text outside the JSON format.
        - Do NOT include markdown syntax like ```json ```.
        - Do NOT deviate from this structure.
        """
    
    def _parse_and_validate_response(self, llm_response: Union[str, dict]) -> Tuple[Optional[dict], bool]:
        parsed_response = None
        
        # --- JSON Validation ---
        if isinstance(llm_response, str):
            try:
                llm_response = json.loads(llm_response.strip().replace('```json', '').replace('```', ''))
            except json.JSONDecodeError as e:
                print(f"[Primary Care] Invalid JSON format: {e}")
                return None, False

        # --- Top-level Structure Validation ---
        # Handle both direct array response and wrapped response
        if isinstance(llm_response, dict) and "specialists" in llm_response:
            specialist_list = llm_response["specialists"]
        elif isinstance(llm_response, list):
            specialist_list = llm_response
        else:
            print("[Primary Care] Response must be either a JSON array or object with 'specialists' key")
            return None, False

        # --- Array Length Validation ---
        if not isinstance(specialist_list, list):
            print("[Primary Care] Specialist data must be in an array")
            return None, False

        if len(specialist_list) < 3 or len(specialist_list) > 5:
            print(f"[Primary Care] Expected 3-5 specialists, got {len(specialist_list)}")
            return None, False

        required_keys = {"specialist", "specialist_role_description", "rationale"}
        validated_specialists = []

        for specialist in specialist_list:
            if not isinstance(specialist, dict):
                print("[Primary Care] Each specialist must be a JSON object")
                return None, False

            # Check for missing or extra keys
            missing_keys = required_keys - set(specialist.keys())
            if missing_keys:
                print(f"[Primary Care] Missing required keys: {missing_keys}")
                return None, False

            # Validate value types
            for key in required_keys:
                if not isinstance(specialist[key], str):
                    print(f"[Primary Care] '{key}' must be a string")
                    return None, False

            # Standardize specialist name formatting
            specialist_name = specialist["specialist"].strip()
            if not specialist_name or specialist_name.lower() == "unknown":
                print("[Primary Care] Specialist name cannot be empty or 'Unknown'")
                return None, False

            validated_specialists.append({
                "specialist": specialist_name,
                "specialist_role_description": specialist["specialist_role_description"].strip(),
                "rationale": specialist["rationale"].strip()
            })

        # --- Build Final Output ---
        parsed_response = {
            "primary_care_choices": validated_specialists,
            "status": "valid",
            "validation_errors": None
        }
        
        return parsed_response, True

    def _retry_with_feedback(self, original_prompt: str, feedback: str) -> str:
        """Sends feedback to AI and asks for a corrected response."""
        retry_prompt = f"""
        [ORIGINAL PROMPT]
        {original_prompt}

        [PREVIOUS RESPONSE WAS INVALID]
        {feedback}

        [REVISED INSTRUCTIONS]
        - Fix the errors mentioned above.
        - Do NOT repeat the same mistakes.
        - Strictly follow the required JSON format.
        - Example of valid format:
        [
            {{
                "specialist": "Cardiologist",
                "specialist_role_description": "...",
                "rationale": "..."
            }}
        ]
        """

        print("\n====== Primary Care Prompt ======")
        print(retry_prompt)

        return self.call_llm(retry_prompt, require_json=True)
    
    def _generate_feedback(self, error_response: Dict) -> str:
        """Generates structured feedback for the AI."""
        if not error_response:
            return "Invalid response format. Please provide a valid JSON array with 3-5 specialists."
        
        if "error" in error_response:
            return f"ERROR: {error_response['error']}\nPlease correct the response."
        else:
            return "Your response did not follow the required JSON format. Ensure:\n1. It is a valid JSON array.\n2. Each entry has 'specialist', 'specialist_role_description', and 'rationale'.\n3. There are 3-5 specialists listed."







