from agents.base_agent import BaseAgent
from typing import Dict, List, Optional, TYPE_CHECKING
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

if TYPE_CHECKING:
    from utils.shared_pool import HistoricalSharedPool

class SpecialistDoctor(BaseAgent):
    # Specialist configurations with role descriptions and task focuses
    SPECIALIST_CONFIGS = {
        # mandatory specialists
        "general_internal_medicine_doctor": {
            "role_description": "You are a General Internal Medicine Doctor, specializing in adult healthcare. Your task is to diagnose and manage a wide range of medical conditions, provide preventive care, and coordinate treatment for complex, chronic illnesses."
        },
        "general_surgeon": {
            "role_description": "You are a General Surgeon, trained to perform a wide range of surgical procedures. Your task is to evaluate the need for surgery (e.g., appendectomy, hernia repair) and manage pre/post-operative care.",
        },
        "pediatrician": {
            "role_description": "You are a Pediatrician, specializing in medical care for infants, children, and adolescents. Your task is to provide age-appropriate diagnoses, vaccinations, and treatment plans.",
        },
        "Obstetrician and gynecologist Specialist": {
            "role_description": "You are an OB/GYN, specializing in women’s reproductive health and pregnancy. Your task is to manage prenatal care, deliveries, and conditions like endometriosis or PCOS.",
        },
        "radiologist": {
            "role_description": "You are a Radiologist, specializing in medical imaging interpretation. Your task is to analyze X-rays, CTs, MRIs, and ultrasounds to diagnose conditions or guide procedures.",
        },
        "neurologist": {
            "role_description": "You are a Neurologist, specializing in nervous system disorders. Your task is to diagnose and treat conditions like epilepsy, stroke, migraines, or Parkinson’s disease.",
        },
        "pathologist": {
            "role_description": "You are a Pathologist, specializing in laboratory medicine. Your task is to diagnose diseases through tissue biopsies, blood tests, and cytology analyses.",
        },
        "pharmacist": {
            "role_description": "You are a Pharmacist, specializing in medication management and pharmacotherapy. Your task is to evaluate prescriptions for safety, efficacy, and potential drug interactions, recommend dosage adjustments based on patient factors (e.g., age, renal function), and provide guidance on proper medication use. You also advise on over-the-counter treatments, monitor for adverse effects, and collaborate with other specialists to optimize therapeutic outcomes."
        },
        # optional
        # Medical Specialties
        "cardiologist": {
            "role_description": "You are a Cardiologist, specializing in heart and cardiovascular system disorders. Your task is to evaluate cardiac symptoms, interpret diagnostic tests, and provide treatment plans for conditions like heart disease, arrhythmias, and hypertension.",
        },
        "endocrinologist": {
            "role_description": "You are an Endocrinologist, specializing in hormonal and metabolic disorders. Your task is to diagnose and manage conditions like diabetes, thyroid disorders, and adrenal gland issues.",
        },
        "gastroenterologist": {
            "role_description": "You are a Gastroenterologist, specializing in digestive system disorders. Your task is to evaluate symptoms related to the stomach, intestines, liver, and pancreas, and recommend treatments for conditions like GERD, IBS, or hepatitis.",
        },
        "hematologist": {
            "role_description": "You are a Hematologist, specializing in blood disorders and cancers. Your task is to diagnose and treat conditions like anemia, clotting disorders, leukemia, and lymphoma.",
        },
        "infectious_disease_specialist": {
            "role_description": "You are an Infectious Disease Specialist, focusing on bacterial, viral, fungal, and parasitic infections. Your task is to identify pathogens, recommend antibiotics/antivirals, and manage complex infections.",
        },
        "nephrologist": {
            "role_description": "You are a Nephrologist, specializing in kidney diseases and hypertension. Your task is to evaluate kidney function, manage dialysis needs, and treat conditions like CKD or kidney stones.",
        },
        "oncologist": {
            "role_description": "You are an Oncologist, specializing in cancer diagnosis and treatment. Your task is to interpret biopsies, recommend chemotherapy/radiation, and coordinate cancer care.",
        },
        "pulmonologist": {
            "role_description": "You are a Pulmonologist, specializing in lung and respiratory disorders. Your task is to evaluate breathing issues, manage COPD/asthma, and interpret imaging/lung function tests.",
        },
        "rheumatologist": {
            "role_description": "You are a Rheumatologist, specializing in autoimmune and joint disorders. Your task is to diagnose and treat conditions like rheumatoid arthritis, lupus, and osteoporosis.",
        },
        # Surgical Specialties
        "cardiothoracic_surgeon": {
            "role_description": "You are a Cardiothoracic Surgeon, specializing in heart, lung, and chest surgeries. Your task is to recommend and perform procedures like bypass surgery or valve replacements.",
        },
        "neurosurgeon": {
            "role_description": "You are a Neurosurgeon, specializing in brain and nervous system surgeries. Your task is to evaluate conditions like tumors, aneurysms, or spinal disorders requiring surgical intervention.",
        },
        "orthopedic_surgeon": {
            "role_description": "You are an Orthopedic Surgeon, focusing on musculoskeletal system surgeries. Your task is to address fractures, joint replacements, or sports injuries requiring surgical repair.",
        },
        "plastic_surgeon": {
            "role_description": "You are a Plastic Surgeon, specializing in reconstructive or cosmetic procedures. Your task is to evaluate needs for trauma reconstruction, burns, or elective enhancements.",
        },
        "trauma_surgeon": {
            "role_description": "You are a Trauma Surgeon, specializing in emergency surgical care for critical injuries. Your task is to stabilize and operate on patients with life-threatening trauma.",
        },
        "vascular_surgeon": {
            "role_description": "You are a Vascular Surgeon, focusing on blood vessel surgeries. Your task is to manage aneurysms, varicose veins, or arterial blockages requiring intervention.",
        },
        # Neurology/Psychiatry
        "psychiatrist": {
            "role_description": "You are a Psychiatrist, specializing in mental health disorders. Your task is to evaluate psychiatric symptoms, prescribe medications, and provide therapy for depression, anxiety, or schizophrenia.",
        },
        # Pediatrics
        "neonatologist": {
            "role_description": "You are a Neonatologist, specializing in newborn and premature infant care. Your task is to manage complications like respiratory distress or infections in neonates.",
        },
        "pediatric_cardiologist": {
            "role_description": "You are a Pediatric Cardiologist, focusing on heart conditions in children. Your task is to diagnose congenital heart defects and manage treatments like surgery or medications.",
        },
        # Emergency Medicine
        "er_physician": {
            "role_description": "You are an ER Physician, specializing in acute care and emergencies. Your task is to stabilize patients with trauma, heart attacks, or severe infections and coordinate further care.",
        },
        # Other Specialties
        "dermatologist": {
            "role_description": "You are a Dermatologist, specializing in skin, hair, and nail disorders. Your task is to diagnose and treat conditions like eczema, psoriasis, or skin cancer.",
        },
        "ophthalmologist": {
            "role_description": "You are an Ophthalmologist, specializing in eye diseases and vision care. Your task is to manage cataracts, glaucoma, or retinal disorders and perform eye surgeries.",
        },
        "ent_specialist": {
            "role_description": "You are an ENT Specialist, focusing on ear, nose, and throat disorders. Your task is to address sinusitis, hearing loss, or tonsillitis, and recommend surgeries if needed.",
        },
        "urologist": {
            "role_description": "You are a Urologist, specializing in urinary tract and male reproductive health. Your task is to manage kidney stones, prostate issues, or incontinence.",
        },
        "allergist": {
            "role_description": "You are an Allergist/Immunologist, specializing in allergies and immune disorders. Your task is to diagnose asthma, food allergies, or autoimmune conditions and recommend treatments.",
        },
        "anesthesiologist": {
            "role_description": "You are an Anesthesiologist, specializing in pain management and sedation. Your task is to administer anesthesia for surgeries and monitor patients’ vital signs during procedures.",
        },
        "physical_medicine_specialist": {
            "role_description": "You are a Physical Medicine Specialist, focusing on rehabilitation. Your task is to manage recovery from injuries, strokes, or chronic pain through non-surgical therapies.",
        },
        # Sub-specialties
        "interventional_cardiologist": {
            "role_description": "You are an Interventional Cardiologist, specializing in minimally invasive heart procedures. Your task is to perform angioplasties, stent placements, or other catheter-based treatments.",
        },
        "electrophysiologist": {
            "role_description": "You are an Electrophysiologist, focusing on heart rhythm disorders. Your task is to diagnose arrhythmias and perform ablations or pacemaker implants.",
        },
        "geriatrician": {
            "role_description": "You are a Geriatrician, specializing in elderly patient care. Your task is to manage age-related conditions like dementia, frailty, or polypharmacy.",
        },
        "sports_medicine_specialist": {
            "role_description": "You are a Sports Medicine Specialist, focusing on athletic injuries. Your task is to diagnose and treat sprains, concussions, or overuse syndromes.",
        },
        "pain_management_specialist": {
            "role_description": "You are a Pain Management Specialist, specializing in chronic pain. Your task is to recommend medications, injections, or therapies for conditions like back pain or neuropathy.",
        },
        "sleep_specialist": {
            "role_description": "You are a Sleep Specialist, focusing on sleep disorders. Your task is to diagnose and treat insomnia, sleep apnea, or narcolepsy.",
        },
        "critical_care_specialist": {
            "role_description": "You are a Critical Care Specialist, managing ICU patients. Your task is to stabilize and treat life-threatening conditions like sepsis or respiratory failure.",
        },
        "palliative_care_specialist": {
            "role_description": "You are a Palliative Care Specialist, focusing on symptom relief for serious illnesses. Your task is to improve quality of life through pain management and supportive care.",
        },
    }

    def __init__(self, specialty: str):
        super().__init__(f"Dr. {specialty}", f"{specialty.capitalize()} Specialist")
        self.specialty = specialty
        self.specialty_config = self.SPECIALIST_CONFIGS.get(
            specialty.lower().replace(" ", "_"),
            {
                "role_description": f"You are a {specialty} Specialist.",
            })
        self.correctKB, self.chainKB = self._load_knowledge_bases()
        # print(f"CorrectKB entries loaded: {len(self.correctKB)}") 
        # print(f"ChainKB entries loaded: {len(self.chainKB)}")
        # print(f"Sample CorrectKB entry: {self.correctKB[0] if self.correctKB else 'Empty!'}")
 
    def perform_task(self, 
        q_num:int,
        question: str, 
        options: str,
        round_num: int,
        history: Dict
        #metadata=None
    ) -> Dict:
        
        #print(f"Round {round_num} - Processing...")
        try:
            
            similar_cases = None

            if round_num > 1:
                # Get history content
                #print(f"\nitems {round_num}: {history}")

                # Get Top 5 similar cases content
                similar_cases = self._retrieve_similar_cases(question)
                #print(f"\nSimilar cases found: {similar_cases}")

            prompt = self._build_prompt(
                question=question,
                options=options,
                round_num=round_num,
                history=history,
                similar_cases=similar_cases
            )
            print("\n====== {} Prompt (ROUND {}) ======".format(self.role, round_num))
            print(prompt)


            response = self._get_llm_response_safe(prompt)
            print("\n====== {} Response (ROUND {}) ======".format(self.role, round_num))
            print(response)

            print("\n====== {} Parsed Response (ROUND {}) ======".format(self.role, round_num))
            parsed_response = self._format_response(response)
            print(parsed_response)

            return parsed_response
            
        except Exception as e:
            print(f"Error in perform_task: {str(e)}") 
            return self._create_error_response(round_num)

    def _load_knowledge_bases(self) -> tuple[List[Dict], List[Dict]]:
        """Load knowledge bases from JSON files"""
        correct_path = "vector_db_storage/CorrectKB.json"
        chain_path = "vector_db_storage/ChainKB.json"

        def load_kb(path: Path) -> List[Dict]:
            try:
                with open(path) as f:
                    data = json.load(f)
                    #print(f"data :{data}")
                    return data
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")  # Add this line
                return []

        return load_kb(correct_path), load_kb(chain_path)

    def _retrieve_similar_cases(self, question: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k similar cases from both KBs"""
        query = f"{question}".strip()

        #print(f"_retrieve_similar_cases,query: {query}")

        def search_kb(kb: List[Dict], kb_name: str) -> List[Dict]:
            if not kb:
                #print("kb is blank")
                return []

            corpus = []
            cases = []
            for case in kb:
                #print(f"\n KB is: {kb}")
                #print(f"\n case is : {case}")
                text = f"{case.get('Question', '')}".strip()
                #print(f"\n text in KB is: {text}")
                if text:
                    corpus.append(text)
                    cases.append(case)

            if not corpus:
                return []

            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                X = vectorizer.fit_transform(corpus + [query])
                similarities = cosine_similarity(X[-1], X[:-1])[0]
                top_indices = np.argsort(similarities)[-k:][::-1]
                return [{
                    'question': cases[i].get('Question', ''),
                    'correct_answer': cases[i].get('Correct Answer', ''),
                    'source': kb_name,
                    'similarity_score': float(similarities[i])
                } for i in top_indices]
            except ValueError:
                return []

        # Get top cases from both KBs
        top_correct = search_kb( self.correctKB,'CorrectKB')
        top_chain = search_kb(self.chainKB, 'ChainKB')
        
        # Combine and return top-k by similarity
        return sorted(top_correct + top_chain, 
                    key=lambda x: x['similarity_score'], 
                    reverse=True)[:k]


    def _build_prompt(
        self,
        question: str,
        options: str,
        round_num: int,
        history: Optional[List[Dict]] = None,
        similar_cases: Optional[List[Dict]] = None) -> str:   

#------------------------Base part prompt--------------------------
        base_parts = [
            f"Present round num:{round_num}",
            "",
            "PATIENT CASE:",
            f"Question: {question}",
            f"Options: {options}",
            "",
            "ROLE DESCRIPTION:",
            self.specialty_config["role_description"]
        ]
        
        if round_num == 1:
            task_parts =[
                "",
                "TASK: ",
                "First, identify the current consultation round number {i}. Each round of discussion follows a consistent Chain-of-Thought (CoT) process.",
                "During each round, you will both call and store data in the shared message pool, referencing previous rounds of discussion.",
                "",
                "CoT PROCESS:",
                "1. Patient Condition Analysis:",
                "Carefully read the patient's description of symptoms, combining their signs, clinical examination, and pregnancy status for a comprehensive analysis.",
                "2. Treatment Option Evaluation:",
                "Based on your professional knowledge, analyze all available treatment options, paying particular attention to drug safety for both the pregnant woman and the fetus.",
                # "3. Call Historical Shared Pool:",
                # "Round {i + 1}: Call the Historical Shared Pool from rounds {i - 1} and {i - 2} to reference the previous two rounds' discussion content for further analysis.",
                # "3. Refine or Confirm Based on Historical pool data:",
                # "Integrate the feedback from the Historical Shared Pool, reassessing your treatment plan. Adjust the selection if necessary; otherwise, confirm the previous choice and explain the rationale.",
                "3. Select Optimal Treatment Plan:",
                "Determine the most appropriate treatment for the patient and explain your decision.",
                "4. Express Conclusion:",
                "Use the following format to express your conclusion:",
                "Choice: {Option ID}: {Option Content}",
                "Example: Choice: {E}: {Nitrofurantoin}",
                "",
                "MDT CONSULTATION FLOW(give answers as these step):"
                "Round {1} Discussion:",
                "Patient Condition Analysis",
                "Treatment Option Evaluation",
                "Select Optimal Treatment Plan",
                "Express Conclusion"
            ]
        elif round_num == 2:
            task_parts =[
                 "",
                "TASK: ",
                "First, identify the current consultation round number {i}. Each round of discussion follows a consistent Chain-of-Thought (CoT) process.",
                "During each round, you will both call and store data in the shared message pool, referencing previous rounds of discussion.",
                "",
                "CoT PROCESS:",
                "1. Patient Condition Analysis:",
                "Carefully read the patient's description of symptoms, combining their signs, clinical examination, and pregnancy status for a comprehensive analysis.",
                "2. Treatment Option Evaluation:",
                "Based on your professional knowledge, analyze all available treatment options, paying particular attention to drug safety for both the pregnant woman and the fetus.",
                # "3. Call Historical Shared Pool:",
                # "Round {i + 1}: Call the Historical Shared Pool from rounds {i - 1} and {i - 2} to reference the previous two rounds' discussion content for further analysis.",
                "3. Refine or Confirm Based on Historical pool data:",
                "Integrate the feedback from the Historical Shared Pool, reassessing your treatment plan. Adjust the selection if necessary; otherwise, confirm the previous choice and explain the rationale.",
                "4. Select Optimal Treatment Plan:",
                "Determine the most appropriate treatment for the patient and explain your decision.",
                "5. Express Conclusion:",
                "Use the following format to express your conclusion:",
                "Choice: {Option ID}: {Option Content}",
                "Example: Choice: {E}: {Nitrofurantoin}",
                "",
                "MDT CONSULTATION FLOW(give answers as these step):"
                # "Round {2} Discussion:",
                "Patient Condition Analysis",
                "Treatment Option Evaluation",
                # "Call Historical Shared Pool from Round {i-1}",
                "Refine or Confirm Based on Historical Pool from last 1 round",
                "Select Optimal Treatment Plan",
                "Express Conclusion"
            ]
        else:
            task_parts =[
                 "",
                "TASK: ",
                "First, identify the current consultation round number {i}. Each round of discussion follows a consistent Chain-of-Thought (CoT) process.",
                "During each round, you will both call and store data in the shared message pool, referencing previous rounds of discussion.",
                "",
                "CoT PROCESS:",
                "1. Patient Condition Analysis:",
                "Carefully read the patient's description of symptoms, combining their signs, clinical examination, and pregnancy status for a comprehensive analysis.",
                "2. Treatment Option Evaluation:",
                "Based on your professional knowledge, analyze all available treatment options, paying particular attention to drug safety for both the pregnant woman and the fetus.",
                # "3. Call Historical Shared Pool:",
                # "Round {i + 1}: Call the Historical Shared Pool from rounds {i - 1} and {i - 2} to reference the previous two rounds' discussion content for further analysis.",
                "3. Refine or Confirm Based on Historical pool data:",
                "Integrate the feedback from the Historical Shared Pool, reassessing your treatment plan. Adjust the selection if necessary; otherwise, confirm the previous choice and explain the rationale.",
                "4. Select Optimal Treatment Plan:",
                "Determine the most appropriate treatment for the patient and explain your decision.",
                "5. Express Conclusion:",
                "Use the following format to express your conclusion:",
                "Choice: {Option ID}: {Option Content}",
                "Example: Choice: {E}: {Nitrofurantoin}",
                "",
                "MDT CONSULTATION FLOW(give answers as these step):"
                # "Round {i + 1} ≥ 3 Discussion:",
                "Patient Condition Analysis",
                "Treatment Option Evaluation",
                # "Call Historical Shared Pool from Round {i-1} and {i-2}",
                # "Store in Historical Shared Pool",
                "Refine or Confirm Based on Historical Pool from last 2 rounds",
                "Select Optimal Treatment Plan",
                "Express Conclusion"
            ]
#------------------------History part prompt--------------------------
        history_parts = []
        if round_num > 1 and history:
            history_parts = ["\nLEAD PHYSICIAN ANALYSIS FROM PREVIOUS ROUNDS:"]
            for item in history:
                independence = "; ".join(item.get('independence', [])) or 'None'
                
                history_parts.extend([
                    f"=== Previous Round ===",
                    f"Consistency: {', '.join(item.get('consistency', [])) or 'None'}",
                    f"Conflict: {', '.join(item.get('conflict', [])) or 'None'}",
                    f"Independence: {independence or 'None'}",
                    f"Integration: {' | '.join(item.get('integration', [])) or 'None'}"
                ])
#------------------------Similar Case part prompt--------------------------       
        # Similar cases section (only for rounds > 1)
        kb_parts = []
        if round_num > 1 and similar_cases:
            kb_parts = ["\nRELEVANT PRIOR CASES:"]
            for i, case in enumerate(similar_cases[:5]):  # Top 5 cases
                #print(f"case: {case}")
                kb_parts.extend([
                    f"=== Case {i+1} [{case['source']}] ===",
                    f"Question: {case['question']}",
                    f"Correct Answer: {case['correct_answer']}",
                    f"Similarity Score: {case['similarity_score']:.2f}\n"
                ])

#------------------------format part prompt--------------------------       
        format_parts=[
            "\nIMPORTANT FORMATTING INSTRUCTIONS:",
            "1. For your final conclusion, you MUST use exactly this format:",
            "   Choice: {Option ID}: {Option Content}",
            "   Example: Choice: {E}: {Nitrofurantoin}",
            "2. Do NOT add any extra text, explanations, or punctuation after the option content",
            "3. The option ID must be a single uppercase letter wrapped in curly braces",
            "4. The option content must be the exact text from the options list, wrapped in curly braces",
            "5. DO NOT USE markdown (**) or any other formatting. "
            "",
        ]

        prompt_parts = base_parts+task_parts+history_parts+kb_parts+format_parts

        example_parts = [
            ""
            "EXAMPLE CASE:",
            "Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She",
            "states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract.",
            "She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C),",
            "blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on",
            "room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus.",
            "Which of the following is the best treatment for this patient?",
            "Options: {\"A\": \"Ampicillin\", \"B\": \"Ceftriaxone\", \"C\": \"Ciprofloxacin\", \"D\": \"Doxycycline\", \"E\": \"Nitrofurantoin\"}",
        ]
        # if round_num == 1:
        #     example_parts += [
        #         "",
        #         "Round {1} Discussion Example:",
        #         "1. Patient Condition Analysis: The patient is 22 weeks pregnant, presenting symptoms consistent with a",
        #         "   urinary tract infection (UTI).",
        #         "2. Treatment Option Evaluation: Considering pregnancy, medication choice should be cautious. Nitrofurantoin",
        #         "   is a relatively safe option for UTI in pregnancy.",
        #         "3. Store in Historical Shared Pool: Store diagnosis and round information.",
        #         "4. Select Optimal Treatment Plan: Nitrofurantoin offers a balance of safety and efficacy for the patient's",
        #         "   condition.",
        #         "5. Express Conclusion: Choice: {E}: {Nitrofurantoin}"
        #     ]
        # elif round_num == 2:
        #     example_parts += [
        #         "",
        #         "Round {2} Discussion Example:",
        #         "1. Patient Condition Analysis: The patient continues to show stable vital signs, consistent with a UTI.",
        #         "2. Treatment Option Evaluation: Nitrofurantoin remains a safe and effective choice.",
        #         "3. Call Historical Shared Pool: Review feedback from other doctors in Round 1.",
        #         "4. Store in Historical Shared Pool: Store the second-round decision and reasoning.",
        #         "5. Refine or Confirm Based on Feedback: Feedback from other doctors supports the selection of Nitrofurantoin, so the choice remains unchanged..",
        #         "6. Express Conclusion: Choice: {E}: {Nitrofurantoin}"
        #     ]
        # else:
        #     example_parts += [
        #         "",
        #         "Round {i + 1} ≥ 3 Discussion Example:",
        #         "1. Patient Condition Analysis: Same as previous rounds; the symptoms remain consistent with a UTI.",
        #         "2. Treatment Option Evaluation: Nitrofurantoin is still the most appropriate choice.",
        #         "3. Call Historical Shared Pool: Reference prior round discussions for further feedback.",
        #         "4. Store in Historical Shared Pool: Store the third-round decision and reasoning.",
        #         "5. Refine or Confirm Based on Feedback: After reviewing previous rounds, Nitrofurantoin remains the confirmed choice.",
        #         "6. Express Conclusion: Choice: {E}: {Nitrofurantoin}"
        #     ]
        
        full_prompt = "\n".join(prompt_parts)
        
        return full_prompt

        
    def _get_llm_response_safe(self, prompt: str, max_retries: int = 10) -> str:
        """Safe LLM communication with retries"""
        for attempt in range(max_retries):
            try:
                response = self.call_llm(prompt, max_tokens=5000)  # Increased tokens
                if response and len(response) > 20:
                    #print(f"\n===Specialist (Raw) Output Attempt {attempt + 1}:===")
                    #print(response)
                    return response
                print(f"Attempt {attempt+1}/{max_retries}: Empty or short response")
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
        raise ValueError("Failed to get valid LLM response")

    def _format_response(self, response: str) -> Dict:
        """Formats and validates the LLM response with robust error recovery"""
        try:
            # Normalize the response first
            normalized = self._normalize_response(response)
            #print(f"\n normalized:{normalized}") 

            reason = self._parse_reasoning(normalized)
            choice = self._parse_choice(normalized)
            #print(f"\n reason:{reason}")
            #print(f"\n choice:{choice}")

            return {
                "Agent_Name": self.role,
                "Reasoning": reason,
                "Choice": choice
            }
        except Exception as e:
            #print(f"Response formatting failed: {str(e)}")
            return {
                "Agent_Name": self.role,
                "Reasoning": "Unable to parse reasoning",
                "Choice": "Unable to parse choice"
            }

    def _normalize_response(self, text: str) -> str:
        text = text.replace('**\n', '\n').replace('\n**', '\n')
        text = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', text)
 
        return text.strip()
    
    def _parse_reasoning(self, text: str) -> str:

        text = re.sub(r'\r\n', '\n', text)
        patterns = [
            # Pattern 1: Header with colon and immediate text
            r'Optimal Treatment Plan:\s*(.*?)(?=\n\w+:|$)',
            
            # Pattern 2: Header without colon + next line content
            r'Optimal Treatment Plan\n(.+?)(?=\n\w+:|$)',
            
            # Pattern 3: Header with colon + potential multi-line
            r'(?s)Optimal Treatment Plan:\s*(.*?)(?=Express Conclusion|\Z)',
            
            # Pattern 4: Header without colon + potential multi-line 
            r'(?s)Optimal Treatment Plan\n(.*?)(?=Express Conclusion|\Z)',
            
            # Fallback patterns
            r'(?:Recommendation|Analysis)[:\s]*(.*?)(?=\n\w+:|$)',
            r'(?s)(?:Discussion|Evaluation):(.*?)(?:Conclusion|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                reasoning = re.sub(r'\n*Express Conclusion.*$', '', reasoning, flags=re.IGNORECASE)
                reasoning = re.sub(r'\n+$', '', reasoning) 
                return reasoning
        return "Not specified"
    
    def _parse_choice(self, text: str) -> str:
        patterns = [
            r'Choice:\s*\{([A-Z])\}:\s*(\{[^}]+?\}|[^\n]+)', # strict pattern
            r'Choice:\s*(\{[A-Z]\}:\s*\{[^}]+\})',  # Alternative format
            r'\{[A-Z]\}:\s*\{[^}]+\}(?=\s*$)'  # Fallback - choice at end of string
            r'Choice:\s*\{?([A-Z])\}?:\s*([^\n]+)',  # loose pattern
            r'([A-Z])\s*:\s*([^\n]+)'    #Fallback: Find any letter-colon pattern
        ]
        # Pattern 1: Strict format with double braces
        for pattern in patterns:
            strict_match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            #print(f"\n strict_match:{strict_match} \n")

            if strict_match:
                letter = strict_match.group(1).upper()
                content = strict_match.group(2).strip(' {}')
                return f"{{{letter}}}: {{{content}}}"
        
        return "No parsable choice found"

    def _create_error_response(self, round_num: int) -> Dict:
        """Creates an error response that maintains the expected structure"""
        return {
            "specialist": self.role,
            "Reasoning": "Error: Could not generate reason",
            "Choice": "Error: Could not get choice",

        }
    
   
    

        

        
        

