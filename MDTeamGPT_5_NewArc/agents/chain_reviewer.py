import json
import os
import re
from typing import Dict, List, Union, Any,Optional
from datetime import datetime
from utils.vector_db import VectorDatabase
from agents.base_agent import BaseAgent

class ChainOfThoughtReviewer(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Nan","ChainReviewer")
        self.correct_kb = VectorDatabase("CorrectKB")
        self.chain_kb = VectorDatabase("ChainKB")
        os.makedirs("vector_db_storage", exist_ok=True)
        self.role_description = """You are a Chain-of-Thought Reviewer, responsible for extracting detailed chains of thought from each doctor’s diagnostic process. 
        Your task is to analyze and record the doctors’ thought processes (information from the Historical Shared Pool) and store this information in a vector database for future analysis and retrieval. 
        You will process experience based on whether the consultation outcome is correct or incorrect,storing the relevant information for both correct and incorrect answers."""

    def perform_task(
            self, 
            final_opinion: str,
            official_answer: str, 
            is_correct: bool,
            question: str,
            options: str,  # Changed from Dict to str
            historical_pool: Dict,  # Fixed capitalization
            metadata: Dict) -> Dict:
        
        prompt = self._build_prompt(
            question=question,
            options=options,
            correct_answer=official_answer,
            final_answer=final_opinion,
            is_correct=is_correct,
            history=historical_pool
        )
        print("\n=== Chain_review Prompt ===")
        print(prompt)

        response = self.call_llm(prompt)
        print("\n=== Chain_review Raw Response ===")
        print(response)

        print("\n=== Chain_review Parsed Response ===")
        storage_data =self._KB_parse_llm_response(is_correct,response)
        print(storage_data)

        if is_correct:
            self._store_correct_answer(storage_data)
        else:
            self._store_chain_of_thought(storage_data)

        return storage_data


    def _build_prompt(self,
            question:str,
            options:str,
            correct_answer:str,
            final_answer:str,
            is_correct:str,
            history:Dict
            ) -> str:
        """Use LLM prompt to perform all analysis steps"""

        # Step 1: change the data in historical pool into text 
        history_txt =self._history_retrieve(history,is_correct)
        #print(f"history retrieve: {history_txt}")

#------------------------Base part prompt--------------------------
        right_parts = [
            "ROLE DESCRIPTION:",
            "You are a Chain-of-Thought Reviewer, responsible for extracting detailed chains of thought from discussions of all specialists' diagnosis. ",
            "Your task is to analyze the specialists thought processes.",
            "",
            "TASK: ",
            "1: Identify the Chain of Thought",
            "Extract the key reasoning steps from the specialists discussions and decision-making process,including their initial assumptions, analysis process, and final conclusions.",
            "All specialists discussion will be provided via Historical_pool data.",
            "2: Structuring the Chain of Thought",
            "Organize these steps in logical order to form a clear chain of thought, ensuring each step shows how the doctor progressed from one stage to the next.",
            "3: Store the Correct Answer",
            "If the consultation outcome is correct, I need content in JSON format to store the extracted and recorded chain of thought.",
            "The content format in JSON should be as follows:",
            "     {",
            '       "Question": [{...}],',
            '       "Answer": [{{Option ID}: {Option Content}}],',
            '       "Summary": [{summary should be the answer in Step 2: Structuring the Chain of Thought...}],',
            "     }",
            "",
            ]
        
        wrong_parts = [
            "ROLE DESCRIPTION:",
            "You are a Chain-of-Thought Reviewer, responsible for extracting detailed chains of thought from discussions of all specialists' diagnosis. ",
            "Your task is to analyze and record the specialists thought processes.",
            "",
            "TASK: ",
            "1: Identify the Chain of Thought",
            "Extract the key reasoning steps from the doctor’s discussions and decision-making process,including their initial assumptions, analysis process, and final conclusions.",
            "All doctor's discussion will be provided via Historical_pool data.",
            "2: Structuring the Chain of Thought",
            "Organize these steps in logical order to form a clear chain of thought, ensuring each step shows how the doctor progressed from one stage to the next.",
            "3: Analyze the Mistake",
            "Identify which steps in the chain of thought might have led to the incorrect diagnosis. Find the root cause, such as faulty assumptions, biased analysis, or overlooked critical information.",
            "4: Document the Reflection Process",
            "Clearly indicate the specific reasons for the error and suggest ways to avoid similar mistakes in the future. Highlight which assumptions or analysis steps were wrong and recommend improvements for future reasoning.",
            "5: Store the Incorrect Answer",
            "If the consultation outcome is incorrect, I need content in JSON format to store the extracted and recorded chain of thought. "
            "The storage format in JSON should be as follows:",
            "     {",
            '       "Question": [{...}],',
            '       "Correct Answer": [{{Option ID}: {Option Content}}],',
            '       "Initial Hypothesis": [{...}],',
            '       "Analysis Process": [{...}],',
            '       "Final Conclusion": [{{Option ID}: {Option Content}}],',
            '       "Error Reflection": [{...}],',
            "     }",
            ""
            ]

        format_prompt=[
            "",
            "Format Requirement:"
            "All answer for storage, shall be key with value instead of just value. "
            "",
        ]
        
        case_prompt = [
            "",
            f"Historical_pool:{history_txt}"
            "",
            f"Official_correct_answer:{correct_answer}",
            "",
            f"agents_answer:{final_answer}",
            "",
            f"Is_correct:{is_correct}",
            "",
        ]

        if is_correct:
            prompt_parts=right_parts+case_prompt
        else:
            prompt_parts=wrong_parts+case_prompt

        return "\n".join(prompt_parts)
    
        
    def _history_retrieve(self,history: Dict,is_correct:str) -> str:
        
        # Extract metadata and round information
        metadata = history.get('metadata', {})
        question = metadata.get('question', '')
        round1 = history.get('round', {}).get(1, {})
        specialist_opinions = round1.get('specialist_opinions', [])
        lead_physician = round1.get('lead_physician_analysis', {})

        history_txt = "\n"
        history_txt += "Patient Case Summary\n"
        history_txt += "Question\n"
        history_txt += f"{question}\n"

        round_numbers = []
        for key in history.keys():
            if isinstance(key, str) and key.startswith('round '):
                try:
                    round_num = int(key.split()[1])  # Extract number from "round X"
                    round_numbers.append(round_num)
                except (IndexError, ValueError):
                    continue
        
        # If no rounds found, use round 1 as default
        if not round_numbers:
            round_numbers = [1]
        
        if is_correct:
            # For correct cases, use only the last round
            last_round_num = max(round_numbers)
            round_key = f"round {last_round_num}"
            round_data = history.get(round_key, {})
            
            specialist_opinions = round_data.get('specialist_opinions', [])
            lead_physician = round_data.get('Lead_Physician_Opinion', {})
            
            # Add only last round specialist opinions
            if specialist_opinions:
                history_txt += "Final Specialist Consensus\n"
                for opinion in specialist_opinions:
                    history_txt += (
                        f"{opinion['Agent_Name']}: {opinion['Reasoning']} "
                        f"Conclusion: {opinion['Choice']}\n\n"
                    )
            
            # Add last round lead physician analysis
            if lead_physician:
                history_txt += "### Final Lead Physician Analysis\n"
                if 'consistency' in lead_physician:
                    history_txt += f"Consistency: {', '.join(lead_physician.get('consistency', []))}\n"
                if 'conflict' in lead_physician:
                    history_txt += f"Conflict: {' '.join(lead_physician.get('conflict', []))}\n"
                if 'independence' in lead_physician:
                    history_txt += f"independence: {', '.join(lead_physician.get('independence', []))}\n"
                if 'integration' in lead_physician:
                    history_txt += f"Integration: {' '.join(lead_physician.get('integration', []))}\n"
                # Also include direct opinion if available
                if 'Reasoning' in lead_physician:
                    history_txt += f"Final Analysis: {lead_physician.get('Reasoning', '')}\n"
        else:
            history_txt += "### Specialist Consensus\n"
            for opinion in specialist_opinions:
                history_txt += (
                    f"{opinion['Agent_Name']}: {opinion['Reasoning']} "
                    f"Conclusion: {opinion['Choice']}\n\n"
                )
            
            # Add lead physician analysis
            if lead_physician:
                history_txt += "### Lead Physician Analysis\n"
                history_txt += f"Consistency: {', '.join(lead_physician.get('consistency', []))}\n"
                history_txt += f"Conflict: {' '.join(lead_physician.get('Conflict', []))}\n"
                history_txt += f"independence: {', '.join(lead_physician.get('independence', []))}\n"
                history_txt += f"Integration: {' '.join(lead_physician.get('integration', []))}\n"
        
        return history_txt
    

    def _load_existing_data(self, file_path: str) -> List[Dict]:
        """Safely load existing data or return empty list"""
        if os.path.exists(file_path):
            #print(f"os path exist: {file_path} \n ")
            try:
                with open(file_path, "r") as f:
                    raw_content = f.read()
                    #print(f"[DEBUG] Raw file content (first 200 chars): {raw_content[:200]}...")
            
                    if not raw_content.strip():
                        #print("[DEBUG] File is empty")
                        return []
            
                # Now try parsing
                with open(file_path, "r") as f:
                    data = json.load(f)
                    #print(f"[DEBUG] Parsed JSON data: {data}")
                    return data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, IOError):
                print(f"[WARNING] Corrupted {file_path}, starting fresh")
        return []


    def _store_correct_answer(self, data: Dict) -> None:
        file_path = "vector_db_storage/CorrectKB.json"

        # 1. Check for errors in input data
        if "error" in data or not data.get("Question"):
            print(f"[WARNING] Invalid data - not storing in CorrectKB. Data: {json.dumps(data, indent=2)[:200]}...")
            return
        
        # 2. Load existing data safely
        existing_data = self._load_existing_data(file_path)
        #print(f"\n exist data: {existing_data}")

        # if os.path.exists(file_path):
        #     with open(file_path, "r") as f:
        #         try:
        #             existing_data = json.load(f)
        #             if not isinstance(existing_data, list):  # Ensure it's a list
        #                 existing_data = [existing_data]
        #         except json.JSONDecodeError:
        #             existing_data = []  # Handle empty/corrupted file
        
        # 3. Append new data (only if valid)
        clean_data = {
            "Question": data.get("Question"),
            "Answer": data.get("Answer"),
            "Summary": data.get("Summary")
        }

        existing_data.append(clean_data)

        #print(f"data to store: {existing_data}")

        # 4. Atomic write with backup
        temp_path = f"{file_path}.tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            os.replace(temp_path, file_path)
            #print(f"[SUCCESS] Appended case to CorrectKB (Total: {len(existing_data)})")
            
            # # 5. Vector DB storage
            # try:
            #     self.correct_kb.store(clean_data)
            #     print(f"\n [SUCCESS] Vector DB storage complete: {clean_data}")
            # except Exception as e:
            #     print(f"[ERROR] Vector DB storage failed: {str(e)}")
                
        except Exception as e:
            print(f"[CRITICAL] Failed to save CorrectKB: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _store_chain_of_thought(self, data: Dict) -> None:
        file_path = "vector_db_storage/ChainKB.json"
        
        # 1. Check for errors in input data
        if "error" in data or not data.get("Question"):
            print(f"[WARNING] Invalid data - not storing in ChainKB. Data: {json.dumps(data, indent=2)[:200]}...")
            return
    
        # 2. Load existing data safely
        existing_data = self._load_existing_data(file_path)
        #print(f"\n exist data: {existing_data}")

        # 3. Append new data (only if valid)
        clean_data = {
            "Question": data.get("Question"),
            "Correct Answer": data.get("Correct Answer"),
            "Initial Hypothesis": data.get("Initial Hypothesis"),
            "Analysis Process": data.get("Analysis Process"),
            "Final Conclusion": data.get("Final Conclusion"),
            "Error Reflection": data.get("Error Reflection"),
        }
        existing_data.append(clean_data)

        #print(f"data to store: {existing_data}")

        # 4. Atomic write with backup
        temp_path = f"{file_path}.tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            os.replace(temp_path, file_path)
            #print(f"[SUCCESS] Appended case to ChainKB (Total: {len(existing_data)})")
            
            # # 5. Vector DB storage
            # try:
            #     self.chain_kb.store(clean_data)
            #     print(f"\n [SUCCESS] Vector DB storage complete: {clean_data}")
            # except Exception as e:
            #     print(f"[ERROR] Vector DB storage failed: {str(e)}")
                
        except Exception as e:
            print(f"[CRITICAL] Failed to save ChainKB: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    

    def _KB_parse_llm_response(self, is_correct: bool, response: str) -> Dict:
        """Parse LLM response with enhanced error handling for both correct/incorrect cases."""
        def clean_and_parse(json_str: str) -> Dict:
            """Helper to clean and parse problematic JSON structures"""
            # First fix common array formatting issues
            json_str = re.sub(r'\["([^"]+)"\s*,\s*"([^"]+)"\]', r'["\1", "\2"]', json_str)
            json_str = re.sub(r'\[\s*\{([^}]+)\}\s*,\s*\{([^}]+)\}\s*\]', r'[{\1}, {\2}]', json_str)
            
            # Fix unescaped quotes in keys
            json_str = re.sub(r'([{,]\s*)"([^"]+)"\s*:', r'\1"\2":', json_str)
            
            # Remove trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try wrapping unquoted values in arrays
                json_str = re.sub(r':\s*([^"{}\[\],\s]+)(?=[,\]}])', r': "\1"', json_str)
                return json.loads(json_str)

        try:
            # Extract JSON portion (more robust extraction)
            json_str = None
            code_blocks = re.findall(r'```(?:json)?(.*?)```', response, re.DOTALL)
            if code_blocks:
                json_str = code_blocks[0].strip()
            
            if not json_str:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)

            if json_str:
                # Special handling for incorrect answer format
                if not is_correct:
                    # Fix malformed arrays in Error Reflection
                    json_str = re.sub(
                        r'"Error Reflection": \[(\{.*?\}|"[^"]*")(?:\s*,\s*(\{.*?\}|"[^"]*"))*\]',
                        self._fix_error_reflection_array,
                        json_str
                    )
                
                parsed_data = clean_and_parse(json_str)
                #print(f"\n not correct, parsed_data: {parsed_data}")
                #print(f"\n validate parsed_data: {self._validate_parsed_data(is_correct, parsed_data)}")
                return self._validate_parsed_data(is_correct, parsed_data)

            #print(f"\n correct,parse text: {self._parse_text_response(is_correct, response)} ")   
            return self._parse_text_response(is_correct, response)
            
        except Exception as e:
            print(f"Error parsing response: {e}\nResponse: {response[:200]}...")
            return self._parse_text_response(is_correct, response)

    def _fix_error_reflection_array(self, match: re.Match) -> str:
        """Helper to properly format Error Reflection array content"""
        items = []
        for item in match.group(0)[20:-1].split(','):  # Skip "Error Reflection": [ and ]
            item = item.strip()
            if item.startswith('{') and item.endswith('}'):
                items.append(item)
            else:
                items.append(f'{{"text": {item}}}')
        return f'"Error Reflection": [{", ".join(items)}]'

    def _validate_parsed_data(self, is_correct: bool, parsed_data: Dict) -> Dict:
        """Enhanced validation with better array handling"""
        # Standardize Question format
        if isinstance(parsed_data.get("Question"), list):
            parsed_data["Question"] = " ".join(
                q.get("text", str(q)) if isinstance(q, dict) else str(q) 
                for q in parsed_data["Question"]
            )
        
        if is_correct:
            # Handle Answer formatting
            if isinstance(parsed_data.get("Answer"), list):
                answers = []
                for a in parsed_data["Answer"]:
                    if isinstance(a, dict):
                        answers.append(a.get("Option Content", str(a)))
                    else:
                        answers.append(str(a))
                parsed_data["Answer"] = " ".join(answers)
        else:
            # Special handling for incorrect answer structure
            if isinstance(parsed_data.get("Error Reflection"), list):
                reflections = []
                for item in parsed_data["Error Reflection"]:
                    if isinstance(item, str):
                        reflections.append({"text": item})
                    elif isinstance(item, dict):
                        reflections.append(item)
                parsed_data["Error Reflection"] = reflections if reflections else {}
                
            # Ensure all fields exist
            parsed_data.setdefault("Correct Answer", "")
            parsed_data.setdefault("Initial Hypothesis", {})
            parsed_data.setdefault("Analysis Process", {})
            parsed_data.setdefault("Final Conclusion", "")
            parsed_data.setdefault("Error Reflection", {})
        
        return parsed_data
        
    def _parse_text_response(self, is_correct: bool, response: str) -> Dict:
        """Fallback method to parse text response when JSON parsing fails.
        
        Args:
            is_correct: Whether the response is for a correct answer
            response: The text response to parse
            
        Returns:
            Dictionary containing parsed data
        """
        storage_data = {
            "error": "Failed to parse JSON, using text fallback",
            "raw_response": response
        }
        
        if is_correct:
            # Extract information from text for correct answer
            question = self._extract_between_markers(response, "Question:", ["Answer:", "Step 2:"]) or ""
            answer = self._extract_between_markers(response, "Answer:", ["Summary:", "Explanation:"]) or ""
            summary = self._extract_after_marker(response, "Summary:") or self._extract_after_marker(response, "Explanation:") or ""
            
            storage_data.update({
                "Question": question,
                "Answer": answer,
                "Summary": summary
            })
        else:
            # Extract information from text for incorrect answer
            question = self._extract_between_markers(response, "Question:", ["Correct Answer:", "Step 2:"]) or ""
            correct_answer = self._extract_between_markers(response, "Correct Answer:", ["Initial Hypothesis:", "Analysis:"]) or ""
            initial_hypothesis = self._extract_between_markers(response, "Initial Hypothesis:", "Analysis:") or ""
            analysis = self._extract_between_markers(response, "Analysis:", "Final Conclusion:") or ""
            final_conclusion = self._extract_after_marker(response, "Final Conclusion:") or ""
            error_reflection = self._extract_after_marker(response, "Error Reflection:") or ""
            
            storage_data.update({
                "Question": question,
                "Correct Answer": correct_answer,
                "Initial Hypothesis": {"text": initial_hypothesis} if initial_hypothesis else {},
                "Analysis Process": {"text": analysis} if analysis else {},
                "Final Conclusion": final_conclusion,
                "Error Reflection": {"text": error_reflection} if error_reflection else {}
            })
        
        return storage_data

    def _extract_between_markers(self, text: str, start_marker: str, end_markers: Union[str, List[str]]) -> str:
        """Helper to extract text between start marker and first occurrence of any end marker.
        
        Args:
            text: Text to search in
            start_marker: Marker to start extraction after
            end_markers: Single marker or list of markers to end extraction before
            
        Returns:
            Extracted text or empty string if markers not found
        """
        if isinstance(end_markers, str):
            end_markers = [end_markers]
        
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return ""
        
        start_idx += len(start_marker)
        
        # Find the earliest end marker
        earliest_end = len(text)
        for marker in end_markers:
            end_idx = text.find(marker, start_idx)
            if end_idx != -1 and end_idx < earliest_end:
                earliest_end = end_idx
        
        if earliest_end == len(text):
            return text[start_idx:].strip()
        return text[start_idx:earliest_end].strip()

    def _extract_after_marker(self, text: str, marker: str) -> str:
        """Helper to extract text after a marker until end of string or next section.
        
        Args:
            text: Text to search in
            marker: Marker to start extraction after
            
        Returns:
            Extracted text or empty string if marker not found
        """
        start_idx = text.find(marker)
        if start_idx == -1:
            return ""
        
        start_idx += len(marker)
        return text[start_idx:].strip()