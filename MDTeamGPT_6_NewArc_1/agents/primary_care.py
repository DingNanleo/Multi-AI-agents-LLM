from agents.base_agent import BaseAgent
from typing import Dict, List, Optional, Union, Tuple
from utils.shared_pool import HistoricalSharedPool
from datetime import datetime
import json

class PrimaryCareDoctor(BaseAgent):

    APPROVED_SPECIALTIES = {
        "Emergency",
        "Cardiology", 
        "Trauma Surgery",
        "Pediatrics",
        "Toxicology",

        "Neurology",
        "Pulmonology",
        "Gastroenterology",
        "Infectious Disease",
        "Orthopedics",

        "Obstetrics",
        "Psychiatry",
        "Radiology",
        "Anesthesiology",
        "Nephrology",
        
        "Hematology",
        "Endocrinology",
        "Rheumatology",
        "Dermatology",
        "General Medicine",

        "Urology"
    }

    def __init__(self, max_retries: int = 10,historical_pool = None):
        super().__init__("Dr. Smith", "Primary Care Physician")
        self.max_retries = max_retries  
        self.historical_pool = historical_pool 
        self.specialty_descriptions = self._init_specialty_descriptions()

    def _init_specialty_descriptions(self) -> Dict[str, str]:
        """Predefined role descriptions for each specialty"""
        return {
        "Emergency": "Manages acute life-threatening conditions and trauma cases",
        "Cardiology": "Specializes in heart conditions and cardiovascular diseases",
        "Trauma Surgery": "Handles critical injuries requiring immediate surgical intervention",
        "Pediatrics": "Provides medical care for infants, children, and adolescents",
        "Toxicology": "Focuses on poisoning, overdose, and chemical exposure management",
        "Neurology": "Diagnoses and treats disorders of the nervous system (brain, spine, nerves)",
        "Pulmonology": "Specializes in respiratory diseases and conditions (e.g., COPD, asthma)",
        "Gastroenterology": "Manages digestive system disorders (e.g., IBS, liver disease)",
        "Infectious Disease": "Treats complex infections (bacterial, viral, fungal, parasitic)",
        "Orthopedics": "Deals with musculoskeletal injuries and conditions (bones, joints, ligaments)",
        "Obstetrics": "Focuses on pregnancy, childbirth, and postpartum care",
        "Psychiatry": "Diagnoses and treats mental health and behavioral disorders",
        "Radiology": "Interprets imaging (X-rays, MRIs, CT scans) for diagnosis and treatment",
        "Anesthesiology": "Manages pain control and sedation during surgical procedures",
        "Nephrology": "Specializes in kidney diseases and dialysis management",
        "Hematology": "Focuses on blood disorders (e.g., anemia, leukemia, clotting issues)",
        "Endocrinology": "Treats hormone-related conditions (e.g., diabetes, thyroid disorders)",
        "Rheumatology": "Manages autoimmune and inflammatory joint diseases (e.g., arthritis)",
        "Dermatology": "Diagnoses and treats skin, hair, and nail conditions",
        "General Medicine": "Provides comprehensive care and coordinates between specialists",
        "Urology": "Specializes in urinary tract and male reproductive system disorders"
        }
    
    def perform_task(self, question: str, options: str) -> Dict:
        print("\n====== Primary Care Prompt ======")
        prompt = self._build_prompt(question, options)
        print(prompt)

        print(f"\n====== Primary Care LLM Response======")
        llm_response = self.call_llm(prompt, require_json=True)
        print(llm_response)

        parsed_response, is_valid, error_details = self._parse_and_validate_response(llm_response)
        
        retry_count = 0
        while not is_valid and retry_count < self.max_retries:
            #feedback = self._generate_feedback(error_details) if parsed_response else "Invalid response format"
            corrected_response = self._retry_with_feedback(prompt, error_details or "Invalid response format")
            print(f"\n====== Primary Care LLM Response======")
            print(corrected_response)
            parsed_response, is_valid, error_details = self._parse_and_validate_response(corrected_response)
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
        You are an expert primary care physician. Analyze the clinical scenario and recommend relevant medical specialists.

        [CASE DETAILS]
        Patient Presentation and Medical Question: {question}
        

        [YOUR TASK]
        Recommend 4-6 specialists from THE EXACT LIST BELOW in priority order (1 = most urgent).
        For EACH specialist you MUST provide:
        1. Must have "General Medicine" (priority=1)
        2. The EXACT specialty name from approved list
        3. Priority number (2-5, no duplicates)
        4. Role description
        5. Clinical rationale

        [APPROVED SPECIALTIES - USE ONLY THESE]
        {self.APPROVED_SPECIALTIES}

        [STRICT FORMAT]
        Your response MUST be a JSON object with this EXACT structure:
        {{
            "specialists": [
                {{
                    "specialist": ""General Medicine"",
                    "priority": 1,
                    "specialist_role_description": "Provides comprehensive care and ...",
                    "rationale": "Mandatory"
                }},
                {{
                    "specialist": "EXACT_NAME_FROM_LIST_ABOVE",
                    "priority": 1,
                    "specialist_role_description": "What this specialist does",
                    "rationale": "Why this specialist is needed"
                }},
                // Add 2-4 more specialists
            ]
        }}
        
        [CRITICAL RULES] 
        - NEVER invent new specialties - ONLY use from the approved list
        - NEVER change specialty names (e.g., don't shorten "Gastroenterology" to "GI")
        - ALWAYS use double quotes for JSON keys/values
        - NEVER include markdown formatting (```json)
        - Priorities must be unique numbers 1-5
        """


    def _parse_and_validate_response(self, llm_response: Union[str, dict]) -> Tuple[Optional[dict], bool, Optional[str]]:
        parsed_response = None
        error_details = None
        
        # JSON Validation
        if isinstance(llm_response, str):
            try:
                llm_response = json.loads(llm_response.strip().replace('```json', '').replace('```', ''))
            except json.JSONDecodeError as e:
                error_details = f"Invalid JSON format: {e}"
                print(f"[Primary Care] {error_details}")
                return None, False, error_details

        # Structure Validation
        if isinstance(llm_response, dict) and "specialists" in llm_response:
            specialist_list = llm_response["specialists"]
        elif isinstance(llm_response, list):
            specialist_list = llm_response
        else:
            error_details = "Response must be either a JSON array or object with 'specialists' key"
            print(f"[Primary Care] {error_details}")
            return None, False, error_details

        # Array Length Validation (4-6 entries: 1 general + 3-5 specialists)
        if not isinstance(specialist_list, list) or len(specialist_list) < 4 or len(specialist_list) > 6:
            error_details = f"Expected 4-6 entries, got {len(specialist_list)}"
            print(f"[Primary Care] {error_details}")
            return None, False, error_details
        
        has_general_medicine = False
        general_medicine_priority = None
        
        for specialist in specialist_list:
            if specialist["specialist"] == "General Medicine":
                has_general_medicine = True
                general_medicine_priority = specialist["priority"]
                break
        
        if not has_general_medicine:
            print("[Primary Care] must have General Medicine.")
            error_details = "Response  must have General Medicine"
            return None, False, error_details
        
        if general_medicine_priority != 1:
            print(f"[Primary Care] General Medicine's priority must be 1 ( present priority is: {general_medicine_priority})")
            error_details = "General Medicine's priority must be 1"
            return None, False, error_details
        
        required_keys = {"specialist", "priority", "specialist_role_description", "rationale"}
        validated_specialists = []
        seen_priorities = set()

        # Add specialty validation
        for specialist in specialist_list:
            spec_name = specialist["specialist"]
            if spec_name not in self.APPROVED_SPECIALTIES:
                print(f"[Primary Care] Invalid specialty: {spec_name}. Must be one of {self.APPROVED_SPECIALTIES}")
                error_details = f"Invalid specialty: {spec_name}"
                return None, False,error_details

            # Check for missing or extra keys
            missing_keys = required_keys - set(specialist.keys())
            if missing_keys:
                print(f"[Primary Care] Missing required keys: {missing_keys}")
                error_details = f"Missing required keys: {missing_keys}"
                return None, False,error_details

            # Validate value types
            for key in required_keys:
                if key == "priority":
                    if not isinstance(specialist[key], int):
                        error_details = f"Priority must be integer, got {type(specialist[key])}"
                        print(f"[Primary Care] {error_details}")
                        return None, False, error_details
                    if specialist[key] in seen_priorities:
                        error_details = f"Duplicate priority: {specialist[key]}"
                        print(f"[Primary Care] {error_details}")
                        return None, False, error_details
                    seen_priorities.add(specialist[key])
                elif not isinstance(specialist[key], str):
                    error_details = f"'{key}' must be a string, got {type(specialist[key])}"
                    print(f"[Primary Care] {error_details}")
                    return None, False, error_details

            # Standardize specialist name
            specialist_name = specialist["specialist"].strip()
            if not specialist_name:
                error_details = "Specialist name cannot be empty"
                print(f"[Primary Care] {error_details}")
                return None, False, error_details

            validated_specialists.append({
                "specialist": specialist_name,
                "priority": specialist["priority"],
                "specialist_role_description": specialist["specialist_role_description"].strip(),
                "rationale": specialist["rationale"].strip()
            })

        # Verify priorities are consecutive starting from 1
        expected_priorities = set(range(1, len(specialist_list) + 1))  # Generates {1, 2, 3,...}
        if seen_priorities != expected_priorities:
            error_details = f"Priorities must be consecutive starting from 1, got {seen_priorities}"
            print(f"[Primary Care] {error_details}")
            return None, False, error_details
        
        
        self._store_opinion_history(validated_specialists)

        # Prepare response without priority
        response_specialists = []
        for specialist in validated_specialists:
            response_specialists.append({
                "specialist": specialist["specialist"],
                "specialist_role_description": specialist["specialist_role_description"],
                "rationale": specialist["rationale"]
            })

        parsed_response = {
            "primary_care_choices": response_specialists
        }

        return parsed_response, True, None

    def _store_opinion_history(self, specialists: List[Dict]):
        """Store all specialists' opinions together in historical pool"""
        if not self.historical_pool:
            print("[PrimaryCare] No historical pool available - skipping storage")
            return
            
        try:
            # Create a single history entry containing all specialists
            history_entry = {
                "Primary_care_choosed_specialists": [
                    {
                        "specialist": spec["specialist"],
                        "priority": spec["priority"],
                        "description": spec["specialist_role_description"],
                        "rationale": spec["rationale"]
                    } for spec in specialists
                ],
                "timestamp": datetime.now().isoformat()
            }
            self.historical_pool.add_statements(history_entry)
            
        except Exception as e:
            print(f"[PrimaryCare] Failed to store specialist opinions: {str(e)}")
            import traceback
            traceback.print_exc()


    def _retry_with_feedback(self, original_prompt: str, feedback: str) -> str:
        retry_prompt = f"""
        [ORIGINAL PROMPT]
        {original_prompt}

        [PREVIOUS RESPONSE WAS INVALID]
        {feedback}

        """

        print("\n====== Primary Care Retry Prompt ======")
        print(retry_prompt)
        return self.call_llm(retry_prompt, require_json=True)
    
    def _generate_feedback(self, error_response: Dict) -> str:
        """Enhanced feedback with specialty guidance"""
        base_feedback = super()._generate_feedback(error_response)
        return (
            f"{base_feedback}\n\n"
            f"REMEMBER:\n"
            f"- You MUST use ONLY these specialties:\n"
            f"{', '.join(sorted(self.APPROVED_SPECIALTIES))}\n"
            f"- General Physician ALWAYS comes first (priority 0)\n"
            f"- Use EXACT specialty names as shown above"
        )














