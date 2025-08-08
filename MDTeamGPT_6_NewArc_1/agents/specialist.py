from agents.base_agent import BaseAgent
from typing import List, Dict, Optional,Tuple,Union
import json
from collections import Counter
from utils.shared_pool import HistoricalSharedPool
from datetime import datetime

###### specialist.py featue ########
###### from round 2, specialist speek one by one instead of round by round, and next specialist can see previous specialist speech ########
###### no limitation of 1-2 sentences when specialist speek ########

class SpecialistAgent(BaseAgent):
    def __init__(self, specialist_info: Dict, question: str, options: List[str],retriever=None):
        super().__init__(
            name=specialist_info['specialist'],  # Required by BaseAgent
            role=specialist_info['specialist_role_description']  # Required by BaseAgent
        )
        self.specialist_type = specialist_info['specialist']  
        self.role_description = specialist_info['specialist_role_description']
        self.rationale = specialist_info.get('rationale', '')
        self.question = question
        self.options = options
        self.current_opinion = None
        self.retriever = retriever  # Add retriever reference
        self.specialty_keywords = self._get_specialty_keywords(specialist_info['specialist'])
    

    def perform_task(self, *args, **kwargs) -> Dict:
        """Implement abstract method from BaseAgent"""
        return kwargs.get('opinion_data', {})

    def _get_specialty_keywords(self, specialty: str) -> List[str]:
        """Returns search terms for each specialty"""
        keyword_map = {
            "Emergency": ["trauma", "resuscitation", "ER", "critical care", "CPR", "triage", "life-threatening"],
            "Cardiology": ["ECG", "echocardiogram", "heart failure", "arrhythmia", "stent", "myocardial", "hypertension"],
            "Trauma Surgery": ["fracture", "hemorrhage", "gunshot", "amputation", "exploratory laparotomy", "damage control"],
            "Pediatrics": ["child", "neonatal", "vaccine", "growth chart", "ADHD", "pediatrician", "well-baby"],
            "Toxicology": ["overdose", "poison", "antidote", "venom", "heavy metals", "naloxone", "toxidrome"],
            "Neurology": ["stroke", "seizure", "Alzheimer's", "migraine", "EEG", "multiple sclerosis", "neuropathy"],
            "Pulmonology": ["COPD", "asthma", "ventilator", "pneumonia", "bronchoscopy", "pulmonary fibrosis", "ABG"],
            "Gastroenterology": ["endoscopy", "colonoscopy", "IBS", "GERD", "cirrhosis", "Crohn's", "hepatitis"],
            "Infectious Disease": ["antibiotics", "sepsis", "HIV", "tuberculosis", "COVID-19", "antimicrobial", "viral load"],
            "Orthopedics": ["fracture", "ACL", "arthroplasty", "spinal fusion", "carpal tunnel", "dislocation", "cast"],
            "Obstetrics": ["prenatal", "C-section", "placenta", "eclampsia", "ultrasound", "postpartum", "OB/GYN"],
            "Psychiatry": ["depression", "bipolar", "schizophrenia", "SSRI", "psychosis", "anxiety", "therapy"],
            "Radiology": ["X-ray", "MRI", "CT scan", "ultrasound", "fluoroscopy", "mammogram", "contrast"],
            "Anesthesiology": ["intubation", "propofol", "epidural", "general anesthesia", "sedation", "pain management"],
            "Nephrology": ["dialysis", "AKI", "CKD", "kidney stone", "glomerulonephritis", "creatinine", "ESRD"],
            "Hematology": ["anemia", "leukemia", "hemoglobin", "coagulation", "blood transfusion", "lymphoma", "DVT"],
            "Endocrinology": ["diabetes", "insulin", "thyroid", "hormone", "osteoporosis", "metabolic", "HbA1c"],
            "Rheumatology": ["arthritis", "lupus", "gout", "autoimmune", "biologics", "Sjögren's", "inflammation"],
            "Dermatology": ["acne", "biopsy", "melanoma", "eczema", "psoriasis", "rash", "Mohs surgery"],
            "General Medicine": ["primary care", "PCP", "referral", "annual physical", "hypertension", "preventive care"],
            "Urology": ["kidney stone", "BPH", "cystoscopy", "prostate", "UTI", "incontinence", "vasectomy"]
        }
        return keyword_map.get(specialty, [])
    




class SpecialistDoctor:
    def __init__(self, historical_pool=None,retriever=None):
        self.specialists = []
        self.historical_pool = historical_pool 
        self.retriever = retriever 
        self._evidence_cache = {}  # {specialist_type: evidence_list}
    

#============================ Main method to handle specialist consultation process ======================   
    def perform_task(self, primary_care_response: Dict, question: str, options: List[str]) -> Dict:
        try:
            #=================== Create specialist agents ======================   
            specialist_infos = primary_care_response.get('primary_care_choices', [])            
            self.specialists = [
                SpecialistAgent(info, question, options,self.retriever)
                for info in specialist_infos
            ]
            print(f"\nConsulting with {len(self.specialists)} specialists:")
            for spec in self.specialists:
                print(f"{spec.specialist_type} : {spec.role_description}")

            #=====================Start discussion until consensus=============
            final_conclusion = self._start_discussion()
            return final_conclusion
            
        except Exception as e:
            return {
                "error": f"Failed to process specialist referral: {str(e)}",
                "original_response": primary_care_response
            }
       
    def _start_discussion(self, max_rounds: int = 100) -> Dict:
        """Orchestrate the discussion process until consensus is reached"""
        current_opinions = []
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'='*20} Round {round_num} {'='*20}")
            
            if round_num == 1:
                # Initial opinions
                current_opinions = self._get_initial_opinions()
            else:
                # Discussion rounds
                current_opinions = self._conduct_sequential_discussion_round(current_opinions, round_num)

            # Check consensus after each round
            is_consensus, _ = self._check_consensus(current_opinions)
            if is_consensus:
                print("\nConsensus reached! Ending discussion.")
                break
            
            if round_num == max_rounds:
                print(f"\nMaximum rounds ({max_rounds}) reached without consensus.")

        # from result at the end
        final_conclusion = self._form_final_conclusion(current_opinions, is_consensus)
        return {"final_conclusion": final_conclusion}
    
#=====================================================================================================

    def _get_initial_opinions(self) -> List[Dict]:
        """Collect initial opinions from all specialists"""
        opinions = []
        for specialist in self.specialists:
            opinion = self.provide_initial_opinion(specialist)
            opinions.append(opinion)
            specialist.perform_task(opinion_data=opinion)
            self._store_initial_opinion_history(opinion, round_num=1)
        return opinions

    def _get_cached_evidence(self, specialist: SpecialistAgent) -> List:
        """获取缓存的证据（如果存在）"""
        cache_key = f"{specialist.specialist_type}_{specialist.question[:50]}"
        return self._evidence_cache.get(cache_key)
    
    def _store_evidence_cache(self, specialist: SpecialistAgent, evidence: List):
        """存储证据到缓存"""
        cache_key = f"{specialist.specialist_type}_{specialist.question[:50]}"
        self._evidence_cache[cache_key] = evidence

    
    # def _format_evidence(self, evidence: List[Dict]) -> str:
    #     if not evidence:
    #         return "No specialty-specific evidence found"
            
    #     evidence_strings = []
    #     for i, e in enumerate(evidence):
    #         evidence_strings.append(
    #             f"Evidence {i+1} (Confidence: {e['confidence']:.2f}, Source: {e['source']}):\n"
    #             f"{e['content']}\n"
    #         )
    #     return "\n".join(evidence_strings)

    def provide_initial_opinion(self, specialist: SpecialistAgent) -> Dict:
        # Get specialty-specific evidence for the first round
        evidence = self.retriever.retrieve_evidence(
            question=specialist.question,
            specialty=specialist.specialist_type,
            k=3
        )
        
        self._store_evidence_cache(specialist, evidence)  # 缓存证据

        # Format evidence for the prompt
        # formatted_evidence = self._format_evidence(evidence)
        
        prompt_base = f"""You are a {specialist.specialist_type} with the following expertise: {specialist.role_description}.
        
        Patient's condition rationale for seeing you: {specialist.rationale}
        
        Medical Question: {specialist.question}
        
        Answer Options: {specialist.options}

        [RELEVANT MEDICAL EVIDENCE]
        {evidence if evidence else "No specialty-specific evidence found"}

        IMPORTANT: You MUST respond in this EXACT JSON format:
        {{
            "selected_option": {{
                "key": "A", 
                "value": "Example Diagnosis"
            }},  // MUST match one of: {specialist.options}
            "reasoning": "Your clinical reasoning here"
        }}
        """
        
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            prompt = prompt_base
            if retry_count > 0:
                prompt += "\n\nYour previous response did not meet the required JSON format. Please try again."
            print(f"\n====== {specialist.specialist_type} Initial Attempt {retry_count+1} Prompt ======")
            print(prompt)


            llm_response = specialist.call_llm(prompt, require_json=True)
            # Handle API errors
            if isinstance(llm_response, str) and '"api_error": true' in llm_response:
                print(f"API Error: {llm_response}")
                retry_count += 1
                continue
            print(f"\n====== {specialist.specialist_type} Initial Attempt {retry_count+1} LLM Response ======")
            print(llm_response)
            
            parsed_response = self._parse_and_validate_initial_response(
                llm_response, 
                specialist.specialist_type, 
                specialist.options
            )
            print(f"\n====== {specialist.specialist_type} Initial Attempt {retry_count+1} Parsed Response ======")
            print(parsed_response)

            if parsed_response:
                return parsed_response
                
            retry_count += 1

        return {
            "round": 1,
            "specialist": specialist.specialist_type,
            "selected_option": specialist.options[0],
            "reasoning": "Default after retries"
        }

    def _parse_and_validate_initial_response(self, llm_response: Union[str, dict], specialist_type: str, 
                                       options: List) -> Optional[dict]:
        
        # --- JSON Validation ---
        if isinstance(llm_response, str):
            try:
                llm_response = json.loads(llm_response.strip().replace('```json', '').replace('```', ''))
            except json.JSONDecodeError:
                print(f"[{specialist_type}] Invalid JSON format")
                return None

        if not isinstance(llm_response, dict):
            return None

        # --- Field Validation ---
        required = {
            "selected_option": None,  # Checked separately
            "reasoning": str
        }
        
        for field, expected_type in required.items():
            if field not in llm_response:
                print(f"[{specialist_type}] Missing field: {field}")
                return None
            if expected_type and not isinstance(llm_response[field], expected_type):
                print(f"[{specialist_type}] Invalid type for {field}: expected {expected_type}")
                return None

        # --- Option Validation ---
        if not self._is_valid_option(llm_response["selected_option"], options):
            print(f"[{specialist_type}] Invalid option selected")
            return None

        # --- Build Final Output ---
        return {
            "round": 1,
            "specialist": specialist_type,
            "selected_option": llm_response["selected_option"],
            "reasoning": llm_response["reasoning"]
        }

    def _store_initial_opinion_history(self, opinion: Dict, round_num: int):
        if not self.historical_pool:
            return
        try:
            specialist_name = opinion.get('specialist', 'unknown').replace(" ", "_")
            history_key = f"round_{round_num}_{specialist_name}"
            
            history_entry = {
                history_key: {  # This ensures each specialist gets unique storage
                    "specialist": opinion.get('specialist'),
                    "selected_option": opinion.get('selected_option'),
                    "reasoning": opinion.get('reasoning'),
                    "timestamp": datetime.now().isoformat()  # Add timestamp for tracking
                }
            }
            self.historical_pool.add_statements(history_entry)
        except Exception as e:
            print(f"Failed to store opinion in history: {str(e)}")

#=====================================================================================================
  
    def _conduct_sequential_discussion_round(self, previous_opinions: List[Dict], round_num: int) -> List[Dict]:
        """Conduct discussion where specialists speak one by one, seeing previous responses"""
        updated_opinions = []
        discussion_summary = []

        for i, (specialist, prev_opinion) in enumerate(zip(self.specialists, previous_opinions)):
            print(f"\n====== {specialist.specialist_type} Speaking (Round {round_num}) ======")
            
            # Create summary of what's been said so far in this round
            current_round_summary = "\n".join(discussion_summary)
            previous_round_summary=self._create_discussion_summary(previous_opinions, round_num)
            
            # Get the updated opinion based on previous responses in this round
            updated_opinion = self._get_sequential_opinion(
                specialist=specialist,
                previous_round_summary=previous_round_summary,
                current_round_summary=current_round_summary,
                prev_opinion=prev_opinion,
                round_num=round_num
            )
            
            updated_opinions.append(updated_opinion)
            specialist.perform_task(opinion_data=updated_opinion)
            self._store_discussion_opinion_history(updated_opinion, round_num)
            
            # Add this specialist's response to the current round summary
            if updated_opinion["decision_type"] == "DEFEND":
                discussion_summary.append(
                    f"{updated_opinion['specialist']}: {updated_opinion['selected_option']} "
                    f"(Strengths: {updated_opinion['your_position_strengths']}) "
                    f"(Opposition Weaknesses: {updated_opinion['others_weaknesses']})"
                )
            else:  # ADJUST
                discussion_summary.append(
                    f"{updated_opinion['specialist']}: {updated_opinion['selected_option']} "
                    f"(Old Weaknesses: {updated_opinion['old_position_weaknesses']}) "
                    f"(New Insights: {updated_opinion['new_insights']})"
                )
        
        return updated_opinions

    def _get_sequential_opinion(self, specialist, previous_round_summary, current_round_summary, prev_opinion, round_num) -> Dict:
        # Get specialty-specific evidence
        evidence = self._get_cached_evidence(specialist)
        if not evidence:
            evidence = "No specialty-specific evidence found"
        #formatted_evidence = self._format_evidence(evidence)

        prompt = f"""
        [IMPORTANT CONTEXT]
        Since no consensus was reached in Round {round_num-1}, you must carefully reconsider your position.
        You have TWO options:
        1. DEFEND your original answer with strong, specific reasoning
        2. ADJUST your answer if convinced by others' arguments with clear justification

        [SPECIALTY-SPECIFIC EVIDENCE]
        {evidence}

        [DISCUSSION HISTORY - PREVIOUS ROUND ({round_num-1})]
        {previous_round_summary}

        [CURRENT ROUND ({round_num}) - DISCUSSION SO FAR]
        {current_round_summary if current_round_summary else "No other specialists have spoken yet this round"}

        [YOUR PREVIOUS POSITION]
        Option: {prev_opinion['selected_option']}

        [STRATEGIC INSTRUCTIONS]
        As a {specialist.specialist_type}, you MUST:
        - Thoroughly analyze all arguments with critical thinking
        - Either:
        A) DEFEND your position by:
            - Providing specific, evidence-based points supporting your view
            - Directly identifying concrete flaws in opposing arguments
        OR
        B) ADJUST your position by:
            - Stating weeknessess in your original positon
            - Explicitly stating which exact arguments or new insights changed your mind

        [DECISION FRAMEWORK - MUST FOLLOW EXACTLY]
        Respond in this EXACT format:
        {{
            "decision_type": "DEFEND|ADJUST",
            "selected_option": {{"key": "X", "value": "Option Name"}},  // MUST match one of: {specialist.options}
            "reasoning": {{
                /// REQUIRED for DEFEND
                "your_position_strengths": "specific advantages of your position", // Only for DEFEND
                "others_weaknesses": "flaws in other positions",  // Only for DEFEND
                // REQUIRED for ADJUST
                "old_position_weaknesses": "weaknesses in original stance",  // Only for ADJUST
                "new_insights": "oints that changed your view"  // Only for ADJUST
            }},
            "round": {round_num},
            "specialist": "{specialist.specialist_type}"
        }}

        [EXAMPLE RESPONSES]
        DEFEND example:
        {{
            "decision_type": "DEFEND",
            "selected_option": {{"key": "X", "value": "Option Name"}}, 
            "reasoning": {{
                "your_position_strengths": "1. 95% success rate for this procedure. 2. Addresses root cause visible in imaging...",
                "others_weaknesses": "1. Medication has only 60% efficacy. 2. Physical therapy won't fix structural damage..."
            }},
            "round": {round_num},
            "specialist": "Cardiologist"
        }}

        ADJUST example:
        {{
            "decision_type": "ADJUST",
            "selected_option": {{"key": "X", "value": "Option Name"}}, 
            "reasoning": {{
                "old_position_weaknesses": "1. Overestimated surgical safety. 2. Underestimated drug efficacy...",
                "new_insights": "1. New studies show 85% drug success. 2. Patient has high surgical risk factors..."
            }},
            "round": {round_num},
            "specialist": "Cardiologist"
        }}
        """
        if round_num > 2:
            prompt += """
            [BREAKING THE DEADLOCK - FINAL ROUND RULES]
            You MUST resolve this disagreement NOW. Follow these steps STRICTLY:
            1. **Identify the single biggest patient risk** (e.g., smoking → cancer). Ignore lower-priority risks.  
            2. **Which option best addresses that #1 risk?** If another specialist’s choice does this better, YOU MUST ADJUST.  
            3. **If tied**, default to the guideline-backed gold standard (e.g., CT urography for hematuria in smokers).  

            [MANDATORY RESPONSE CHANGES]  
            You CANNOT defend your position unless:  
            - Your option is *superior* for the #1 risk, AND  
            - At least 2 other specialists already agree with you.  
            """

        max_retries = 5
        for attempt in range(max_retries):
            print(f"\n====== {specialist.specialist_type} Round {round_num} Attempt {attempt+1} Prompt ======")
            print(prompt)

            print(f"\n====== {specialist.specialist_type} Round {round_num} Attempt {attempt+1} LLM Response ======")
            llm_response = specialist.call_llm(prompt, require_json=True)
            print(llm_response)
            
            print(f"\n====== {specialist.specialist_type} Round {round_num} Attempt {attempt+1} Parsed Response ======")
            parsed = self._parse_and_validate_response(
                llm_response,
                specialist.specialist_type,
                specialist.options,
                round_num,
                prev_opinion
            )
            print(parsed)
            
            if parsed:
                return parsed
            
        # Fallback if all retries fail
        return {
            **prev_opinion,
            "round": round_num,
            "decision_type": "FALLBACK",
            "reasoning": "Invalid response after retries"
        }

    def _parse_and_validate_response(self, llm_response: Union[str, dict], specialist_type: str, 
                               options: List, round_num: int, prev_opinion: dict) -> Optional[dict]:

        # --- JSON Validation ---
        if isinstance(llm_response, str):
            try:
                llm_response = json.loads(llm_response.strip().replace('```json', '').replace('```', ''))
            except json.JSONDecodeError:
                print(f"[{specialist_type}] Invalid JSON format")
                return None

        if not isinstance(llm_response, dict):
            return None

        # --- Field Validation ---
        required = {
            "decision_type": ["DEFEND", "ADJUST"],
            "selected_option": None,  # Checked separately
            "reasoning": None
        }
        
        for field, allowed in required.items():
            if field not in llm_response:
                print(f"[{specialist_type}] Missing field: {field}")
                return None
            if field == "reasoning":
                if not isinstance(llm_response[field], dict):
                    print(f"[{specialist_type}] Reasoning must be a dictionary")
                    return None
            if allowed and llm_response[field] not in allowed:
                print(f"[{specialist_type}] Invalid value for {field}: {llm_response[field]}")
                return None

        # --- Option Validation ---
        if not self._is_valid_option(llm_response["selected_option"], options):
            print(f"[{specialist_type}] Invalid option selected")
            return None

        # --- Decision Logic Validation ---
        reasoning = llm_response["reasoning"]
        if llm_response["decision_type"] == "DEFEND":
            if not all(f in reasoning for f in ["your_position_strengths", "others_weaknesses"]):
                return None
        elif llm_response["decision_type"] == "ADJUST":
            if not all(f in reasoning for f in ["old_position_weaknesses", "new_insights"]):
                return None

        # --- Build Final Output ---
        return {
            "round": round_num,
            "specialist": specialist_type,
            "selected_option": llm_response["selected_option"],
            "decision_type": llm_response["decision_type"],
            **llm_response["reasoning"]  # Flatten reasoning fields
        }
    
    def _is_valid_option(self, option: Union[str, dict], valid_options: List) -> bool:
        """Checks if an option matches the allowed choices"""
        if isinstance(option, dict):
            return any(
                opt.get("key") == option.get("key") 
                and opt.get("value") == option.get("value")
                for opt in valid_options
                if isinstance(opt, dict)
            )
        return option in valid_options

    def _store_discussion_opinion_history(self, updated_opinions: Dict, round_num: int):
        if not self.historical_pool:
            return
        try:
            specialist_name = updated_opinions.get('specialist', 'unknown').replace(" ", "_")
            history_key = f"round_{round_num}_{specialist_name}"
            history_entry = {
                history_key: {
                    "specialist": updated_opinions.get('specialist'),
                    "selected_option": updated_opinions.get('selected_option'),
                    "decision_type": updated_opinions.get('decision_type')
                }
            }
            # Add reasoning fields based on decision type
            if updated_opinions.get('decision_type') == "DEFEND":
                history_entry[history_key].update({
                    "your_position_strengths": updated_opinions.get('your_position_strengths'),
                    "others_weaknesses": updated_opinions.get('others_weaknesses'),
                    "timestamp": datetime.now().isoformat()
                })
            elif updated_opinions.get('decision_type') == "ADJUST":
                history_entry[history_key].update({
                    "old_position_weaknesses": updated_opinions.get('old_position_weaknesses'),
                    "new_insights": updated_opinions.get('new_insights'),
                    "timestamp": datetime.now().isoformat()
                })

            self.historical_pool.add_statements(history_entry)
        except Exception as e:
            print(f"Failed to store opinion in history: {str(e)}")


    def _check_consensus(self, opinions: List[Dict]) -> Tuple[bool, List[str]]:
        print("\n=========== Checking Consensus ===========")
        if not opinions:
            print("No opinions available - consensus not reached")
            return False,[]
        
        # Extract and normalize all options for comparison
        normalized_options = []
        specialist_choices = []

        for opinion in opinions:
            option = opinion['selected_option']
            # Normalize the option while preserving original structure
            normalized = self._normalize_option(option)
            normalized_options.append(normalized)

            # Create display version for logging
            if isinstance(option, dict):
                display_option = f"{option.get('key')}: {option.get('value', '').strip()}"
            else:
                display_option = str(option).strip()
                
            specialist_choices.append(
                f"{opinion.get('specialist', 'Unknown')} (Round {opinion['round']}): {display_option}"
            )

        # Log all choices
        for choice in specialist_choices:
            print(choice)

        # Check consensus by comparing normalized versions
        unique_options = set(normalized_options)
        is_consensus = len(unique_options) == 1
            
        if is_consensus:
            print(f"Consensus Reached - All specialists agree on: {normalized_options[0]}")
        else:
            print(f"No Consensus - Found {len(set(normalized_options))} different opinions")
        
        return is_consensus, normalized_options
    
    def _normalize_option(self, option) -> str:
        """Normalize an option for consistent comparison while preserving meaning"""
        if isinstance(option, dict):
            # For dict options, create consistent string representation
            normalized_dict = {
                'key': str(option.get('key', '')).strip().upper(),
                'value': str(option.get('value', '')).strip()
            }
            return json.dumps(normalized_dict, sort_keys=True)
        else:
            # For string options, standardize formatting
            return str(option).strip().lower()

    def _form_final_conclusion(self, opinions: List[Dict],is_consensus: bool) -> Dict:
        if not opinions:
            result =  {
                "status": "no_opinions",
                "selected_option": None,
                "reasoning": "No opinions provided",
                "consensus": False,
                "specialists": []
            }

        if is_consensus:
            result =  {
                "status": "consensus is reached",
                "selected_option": opinions[0]['selected_option'],
                "consensus": True,
                "specialists": [op['specialist'] for op in opinions]
            }
        else:
            # No consensus - return all opinions for next round
            result =  {
                "status": "consensus is not reached",
                "selected_option": None,
                "consensus": False,
                "specialists": [op['specialist'] for op in opinions]
            }
        return result   
    
    def _create_discussion_summary(self, opinions,round_num):
        summary_lines = []
        for opinion in opinions:
            if round_num == 2:
                summary_lines.append(
                    f"{opinion['specialist']}: {opinion['selected_option']} "
                    f"(Reasoning: {opinion.get('reasoning', 'N/A')})"
                )
            else:
                if opinion["decision_type"] == "DEFEND":
                    summary_lines.append(
                        f"{opinion['specialist']}: {opinion['selected_option']} "
                        f"(Strengths: {opinion['your_position_strengths']}) "
                        f"(Opposition Weaknesses: {opinion['others_weaknesses']})"
                    )
                else:  # ADJUST
                    summary_lines.append(
                        f"{opinion['specialist']}: {opinion['selected_option']} "
                        f"(Old Weaknesses: {opinion['old_position_weaknesses']}) "
                        f"(New Insights: {opinion['new_insights']})"
                    )
        
        return "\n".join(summary_lines)
    
        
