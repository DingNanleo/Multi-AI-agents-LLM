from agents.base_agent import BaseAgent
from agents.specialist import SpecialistDoctor 
from typing import Dict, List, Union, Optional, TYPE_CHECKING
import json
import re

class PrimaryCareDoctor(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Smith", "Primary Care Physician")
        self.specialists = {    
            "general_internal_medicine":"general_internal_medicine_doctor",
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

            # Pharmacy
            "pharmacy": "Pharmacist",
            
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
        
        
    def perform_task(self, question: str, options: str) -> Dict:
          
        prompt = self._build_prompt(
            question=question,
            options=options
            )
        print("\n====== Primary Care Prompt ======")
        print(prompt)
        
        response = self._get_llm_response_safe(prompt)
        print("\n====== Primary Care Raw Response ======")
        print(response)

        parsed_response = self._parse_response(response)
        print("\n====== Primary Care Parsed Response ======")
        print(parsed_response)

        return parsed_response
            
    
    def _build_prompt(
            self,
            question: str,
            options: str) -> str:
        
#------------------------Base part prompt--------------------------
        base_parts = [
        "ROLE DESCRIPTION:",
        "You are a Primary Care Doctor Agent (triage doctor) responsible for assigning the appropriate doctors to diagnose and treat patients. Each case involves a specific combination of doctors.",  
        "Based on the symptoms and signs of the patient, choose the most suitable combination of doctors. ",
        "The selection should include the following roles: General Internal Medicine Doctor, General Surgeon, Pediatrician,Obstetrician and Gynecologist, Radiologist, Neurologist, Pathologist, Pharmacist, with Radiologist, Pathologist, and Pharmacist being mandatory agents, while other agent roles are assigned based on specific patient conditions.",
        "",
        "PATIENT CASE:",
        f"Question: {question}",
        f"Options: {options}",
        "",
        # "OUTPUT FORMAT:",
        # "Before selecting the roles, you are required to provide reasons for your choice: Why are these doctors selected based on the patient’s symptoms, signs, and history? How will each selected doctor contribute to the diagnosis and treatment of the patient?",
        # "When making your selection, consider all relevant information about the patient to ensure that all potential issues are covered. The output should be in the following format: ", 
        # "Reasoning: .......",
        # "Output Roles: [{agent1}, {agent2}, {agent3}, ...]",
        # "For example:",
        # "Reasoning: .......",
        # "Output Roles: [{OB/GYN}, {Radiologist}, {Pathologist}, {Pharmacist}, ...]",
        #  "",
        # "STRICT OUTPUT REQUIREMENTS:",
        # "1. Select more than 1 specialist,Format MUST be: [{Agent1}, {Agent2}, {Agent3},...] with curly braces,NEVER use quotes around agent names",
        # "2. Do not give me this : **Output Roles:** , I just want Pure Reason and Output Roles.",
        # "3. Mandatory agents must always be included",
        # "",
        "OUTPUT FORMAT:",
        "You must respond ONLY in valid JSON format with the following keys:",
        "1. 'reasoning': A string explaining why these doctors are selected based on the patient’s symptoms and signs.",
        "2. 'specialists': An array of specialist types (use the internal config keys like 'radiology', 'neurology', 'general_internal_medicine_doctor').",
        "",
        "EXAMPLE:",
        "{",
        '  "reasoning": "Chest pain and ST elevation suggest acute coronary syndrome. Cardiologist is needed for evaluation. Radiologist for imaging. Pathologist for blood markers. Pharmacist for medications.",',
        '  "specialists": ["cardiology", "radiology", "pathology", "pharmacy"]',
        "}",
        "",
        "STRICT REQUIREMENTS:",
        "• DO NOT include extra text or markdown — only return pure JSON.",
        "• Use internal config keys, not human-readable names.",
        "• Include all mandatory agents: 'radiology', 'pathology', and 'pharmacy'."
        ] 
#------------------------example part prompt--------------------------
        example_parts=[
        "EXAMPLE CASE:",
        "Patient: 23yo pregnant woman at 22 weeks with urinary burning",
        "Key Findings:",
        "- No costovertebral angle tenderness",
        "- Gravid uterus",
        "- Stable vitals",
        "Reasoning:",
        "- OB/GYN needed for pregnancy-specific UTI concerns",
        "- Radiologist for urinary system imaging",
        "- Pathologist for lab analysis",
        "- Pharmacist for medication safety",
        "Output: [{OB/GYN}, {Radiologist}, {Pathologist}, {Pharmacist}]",
        "",
        "TIPS:",
        "- Carefully analyze all patient factors",
        "- Balance specialties to cover all potential issues",
        "- Provide clear justification for each selected specialist",
        "- Mandatory agents must always be included"
        ]

        prompt_parts=base_parts
        #prompt_parts=base_parts+example_parts

        return "\n".join(prompt_parts)
        
        
    def _get_llm_response_safe(self, prompt: str, max_retries: int = 10) -> str:
        """Safe LLM communication with retries"""
        for attempt in range(max_retries):
            try:
                response = self.call_llm(prompt, max_tokens=1000)  # Increased tokens
                if response and len(response) > 20:
                    return response
                print(f"Attempt {attempt+1}/{max_retries}: Empty or short response")
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
        raise ValueError("Failed to get valid LLM response")
    
        
    def _parse_response(self, response: str) -> dict:
        
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
            "reasoning": ["Failed to parse specialist recommendations"]
        }