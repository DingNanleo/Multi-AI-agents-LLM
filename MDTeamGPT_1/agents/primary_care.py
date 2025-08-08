from agents.base_agent import BaseAgent

class PrimaryCareDoctor(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Smith", "Primary Care Physician")
        self.specialists = {
            # Medical Specialties
            "cardiology": "Cardiologist",
            "endocrinology": "Endocrinologist",
            "gastroenterology": "Gastroenterologist",
            "hematology": "Hematologist",
            "infectious_diseases": "Infectious Disease Specialist",
            "nephrology": "Nephrologist",
            "oncology": "Oncologist",
            "pulmonology": "Pulmonologist",
            "rheumatology": "Rheumatologist",
            
            # Surgical Specialties
            "general_surgery": "General Surgeon",
            "cardiothoracic_surgery": "Cardiothoracic Surgeon",
            "neurosurgery": "Neurosurgeon",
            "orthopedic_surgery": "Orthopedic Surgeon",
            "plastic_surgery": "Plastic Surgeon",
            "trauma_surgery": "Trauma Surgeon",
            "vascular_surgery": "Vascular Surgeon",
            
            # Neurology/Psychiatry
            "neurology": "Neurologist",
            "psychiatry": "Psychiatrist",
            
            # Pediatrics
            "pediatrics": "Pediatrician",
            "neonatology": "Neonatologist",
            "pediatric_cardiology": "Pediatric Cardiologist",
            
            # OB/GYN
            "obstetrics": "Obstetrician",
            "gynecology": "Gynecologist",
            "obstetrics_gynecology": "OB/GYN",
            
            # Emergency Medicine
            "emergency_medicine": "ER Physician",
            
            # Diagnostic Specialties
            "radiology": "Radiologist",
            "pathology": "Pathologist",
            
            # Other Specialties
            "dermatology": "Dermatologist",
            "ophthalmology": "Ophthalmologist",
            "otolaryngology": "ENT Specialist",
            "urology": "Urologist",
            "allergy_immunology": "Allergist",
            "anesthesiology": "Anesthesiologist",
            "physical_medicine_rehab": "Physical Medicine Specialist",
            
            # Sub-specialties
            "interventional_cardiology": "Interventional Cardiologist",
            "cardiac_electrophysiology": "Electrophysiologist",
            "geriatrics": "Geriatrician",
            "sports_medicine": "Sports Medicine Specialist",
            "pain_management": "Pain Management Specialist",
            "sleep_medicine": "Sleep Specialist",
            "critical_care": "Critical Care Specialist",
            "palliative_care": "Hospice and Palliative Care Specialist"

        }
        
    def perform_task(self, patient_background: str, medical_problem: str) -> dict:
        # Truncate long inputs to 300 chars each
        bg_short = (patient_background[:300] + '...') if len(patient_background) > 300 else patient_background
        prob_short = (medical_problem[:300] + '...') if len(medical_problem) > 300 else medical_problem
    
        prompt = f"""As {self.name} ({self.role}), evaluate:
    Patient: {bg_short}
    Problem: {prob_short}

    Which specialists are needed from: {', '.join(self.specialists.values())}

    Respond ONLY with valid JSON containing:
    1. "specialists": array of specialist types
    2. "reasons": brief justification for each (1 sentence)

    Example:
    {{
        "specialists": ["Cardiologist"],
        "reasons": ["Chest pain requires cardiac evaluation"]
    }}"""
        #print(f"\nPrimary Care Prompt: {prompt}")
        try:
            response = self.call_llm(
                prompt, 
                temperature=0.5,
                max_tokens=500  # Reduced from default
            )
            #print(f"\nPrimary Care response: {response}")
            return self._parse_response(response)

    
        except Exception as e:
            print(f"Primary care consultation failed: {str(e)}")
            return {
                "specialists": [],
                "reasons": ["Error in consultation"]
            }
    
    def _parse_response(self, response: str) -> dict:
        import json
        import re
    
        # First try direct JSON parsing
        try:
            # Clean response by removing potential markdown code blocks
            clean_response = re.sub(r'```json|```', '', response).strip()
            return json.loads(clean_response)
        except json.JSONDecodeError:
            pass
    
        # If that fails, try to extract JSON from the response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
        # Final fallback - return empty with error message
        return {
            "specialists": [],
            "reasons": ["Failed to parse specialist recommendations"]
        }