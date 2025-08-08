import json
import os
import re
from typing import Dict, List, Union, Any,Optional
from datetime import datetime
from utils.vector_db import VectorDatabase
from agents.base_agent import BaseAgent

class ChainOfThoughtReviewer(BaseAgent):
    def __init__(self):
        super().__init__("Dr.Reviewer","ChainReviewer")
        self.correct_kb = VectorDatabase("CorrectKB")
        self.chain_kb = VectorDatabase("ChainKB")
        os.makedirs("vector_db_storage", exist_ok=True)
  
    def perform_task(
            self, 
            final_opinion: str,
            official_answer: str, 
            is_correct: bool,
            question: str,
            options: str,  # Changed from Dict to str
            total_history: Dict,  # Fixed capitalization
            metadata: Dict) -> Dict:
        
        prompt = self._build_prompt(
            question=question,
            options=options,
            correct_answer=official_answer,
            final_answer=final_opinion,
            is_correct=is_correct,
            history=total_history
        )
        print("\n=== Chain_review Prompt ===")
        print(prompt)

        print("\n=== Chain_review Raw Response ===")
        response = self.call_llm(prompt)
        print(response)

        print("\n=== Chain_review Parsed Response ===")
        storage_data =self._KB_parse_llm_response(is_correct,response,question,official_answer,final_opinion)
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
        # print(f"history retrieve: {history_txt}")

#------------------------Base part prompt--------------------------
        right_parts = [
            "ROLE DESCRIPTION:",
            "You are a Chain-of-Thought Reviewer, responsible for extracting detailed chains of thought based on all agents' diagnostic process. ",
            "",
            "TASK: ",
            "1: Based on all agents opinions, identify the Chain of Thought and structuring the Chain of Thought into summary and ouput to me in this JSON format:",
            "For CORRECT answers, use EXACTLY this structure:",
            '```json',
            '{',
            '  "Question": "question text",',
            '  "Answer": { "OptionID": "Content"}(Example: Answer: {E}: {Nitrofurantoin}),',
            '  "Summary": "Summary of all agents reasoning"',
            '}',
            '```'

        ]
        
        wrong_parts = [
            "ROLE DESCRIPTION:",
            "You are a Chain-of-Thought Reviewer, responsible for extracting detailed chains of thought,Initial Hypothesis, Analysis Process,and Error Reflection from all agents' diagnostic process,",
            "",
            "TASK: ",
            "1: Based on all agents opinions, identify the Chain of Thought,structure the Chain of Thought into summary and output to me: Initial Hypothesis, Analysis Process,and Error Reflection and ouput the summarized results in this JSON format:",
            "For INCORRECT answers, use EXACTLY this structure:",
            '```json',
            '{',
            '  "Question": "question text",',
            '  "Correct Answer": {"OptionID": "Content"}(Example: Correct Answer: {E}: {Nitrofurantoin}),',
            '  "Initial Hypothesis": "initial hypothesis text",',
            '  "Analysis Process": "analysis step 1","analysis step 2",',
            '  "Final Conclusion": {"OptionID": "Content"}(Example: Final Conclusion: {E}: {Nitrofurantoin}),',
            '  "Error Reflection": "error analysis text"',
            '}',
            '```'
            ]
 
        case_prompt = [
            f"QUESTION: {question}",
            f"Official_correct_answer:{correct_answer}",
            f"Agents_answer:{final_answer}",
            f"Is_correct:{is_correct}",
            f"Historical_pool:{history_txt}"
            "",
        ]

        format_part=[
            "OUTPUT REQUIREMENTS:",
            "1. Respond ONLY with valid JSON wrapped in ```json ``` markers",
            "2. Use double quotes ONLY (no single quotes)",
            "3. All keys must be camelCase without spaces",
            "4. Arrays must use proper JSON syntax (e.g., [\"item1\", \"item2\"])",
            "5. Never use trailing commas",
            "6. Do not include any text outside the JSON block"
        ]
        if is_correct:
            prompt_parts=case_prompt+right_parts+format_part
        else:
            prompt_parts=case_prompt+wrong_parts+format_part

        return "\n".join(prompt_parts)
    
        
    def _history_retrieve(self,history: Dict,is_correct:str) -> str:
  
        metadata = history.get('metadata', {})
        question = metadata.get('question', '')

        # round1 = history.get('round', {}).get(1, {})
        # specialist_opinions = round1.get('specialist_opinions', [])
        # lead_physician = round1.get('lead_physician_analysis', {})
        # #print(f"history: {history}")

        history_txt = "\nMedical Question\n"
        history_txt += f"{question}\n"

        round_numbers = []
        for key in metadata.keys():
            if isinstance(key, str) and key.startswith('round '):
                try:
                    round_num = int(key.split()[1]) 
                    round_numbers.append(round_num)
                except (IndexError, ValueError):
                    continue
        
        if not round_numbers:
            round_numbers = [1]
    
        if is_correct:
             # For correct cases, use only the last round
            last_round_num = max(round_numbers)
            round_key = f"round {last_round_num}"
            round_data = metadata.get(round_key, {})
            
            specialist_opinions = round_data.get('specialist_opinions', [])
            lead_physician = round_data.get('Lead_Physician_Opinion', {})

            if specialist_opinions:
                history_txt += "\nSpecialist Opinions: \n"
                for opinion in specialist_opinions:
                    history_txt += (
                        f"{opinion['Agent_Name']}: {opinion['Reasoning']} "
                        f"Conclusion: {opinion['Choice']}\n"
                    )
            
            # Add last round lead physician analysis
            if lead_physician:
                history_txt += "\nFinal Lead Physician Analysis: \n"
                if 'consistency' in lead_physician:
                    history_txt += f"Consistency: {', '.join(lead_physician.get('consistency', []))}\n"
                if 'conflict' in lead_physician:
                    history_txt += f"Conflict: {' '.join(lead_physician.get('conflict', []))}\n"
                if 'independence' in lead_physician:
                    history_txt += f"independence: {', '.join(lead_physician.get('independence', []))}\n"
                if 'integration' in lead_physician:
                    history_txt += f"Integration: {' '.join(lead_physician.get('integration', []))}\n"
        
        elif not is_correct:
            for round_num in sorted(round_numbers):
                round_key = f"round {round_num}"
                round_data = metadata.get(round_key, {})
                
                specialist_opinions = round_data.get('specialist_opinions', []) or round_data.get('Specialist_Opinions', [])
                lead_physician = round_data.get('Lead_Physician_Opinion', {}) or round_data.get('lead_physician_opinion', {})
                
                history_txt += f"\n=== Round {round_num} ===\n"

                # Add specialist opinions for this round
                if specialist_opinions:
                    history_txt += "\nSpecialist Opinions: \n"
                    for opinion in specialist_opinions:
                        history_txt += (
                            f"{opinion['Agent_Name']}: {opinion['Reasoning']} "
                            f"Conclusion: {opinion['Choice']}\n"
                        )

                if lead_physician:
                    history_txt += "\nFinal Lead Physician Analysis: \n"
                    if 'consistency' in lead_physician:
                        history_txt += f"Consistency: {', '.join(lead_physician.get('consistency', []))}\n"
                    if 'conflict' in lead_physician:
                        history_txt += f"Conflict: {' '.join(lead_physician.get('conflict', []))}\n"
                    if 'independence' in lead_physician:
                        history_txt += f"independence: {', '.join(lead_physician.get('independence', []))}\n"
                    if 'integration' in lead_physician:
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


        clean_data = {
            "Question": data.get("Question") or data.get("question"),
            "Correct Answer": data.get("Answer") or data.get("answer"),
            "Summary": data.get("Summary") or data.get("summary")
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
            "Correct Answer": data.get("Correct Answer") or data.get("Correct_Answer") or data.get("correct answer") ,
            "Initial Hypothesis": data.get("Initial Hypothesis") or data.get("InitialHypothesis") or data.get("Initial_Hypothesis") or data.get("initial hypothesis"),
            "Analysis Process": data.get("Analysis Process") or data.get("AnalysisProcess")or data.get("Analysis_Process") or data.get("analysis process"),
            "Final Conclusion": data.get("Final Conclusion") or data.get("FinalConclusion") or data.get("Final_Conclusion") or data.get("final_conclusion"),
            "Error Reflection": data.get("Error Reflection") or data.get("ErrorReflection")or data.get("Error_Reflection") or data.get("error reflection"),
        } 
        existing_data.append(clean_data)

        #print(f"data to store: {existing_data}")

        # 4. Atomic write with backup
        temp_path = f"{file_path}.tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            os.replace(temp_path, file_path)
                
        except Exception as e:
            print(f"[CRITICAL] Failed to save ChainKB: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    

    def _KB_parse_llm_response(self, is_correct: bool, response: str, question: str,official_answer:str,final_opinion:str) -> Dict:
        """Parse LLM response handling both correct and incorrect answer formats"""
        
        def _extract_json(text: str) -> Optional[str]:
            """Extract JSON string from response text"""
            # First remove any API error messages
            text = re.sub(r'Attempt \d/5 failed\. Retrying in \d+s\.\.\. Error: 429 Client Error.*?\n', '', text)
            
            # Try code blocks first
            code_blocks = re.findall(r'```(?:json)?(.*?)```', text, re.DOTALL)
            if code_blocks:
                return code_blocks[-1].strip()  # Get last code block if multiple
            
            # Try standalone JSON
            json_match = re.search(r'\{[\s\S]*\}', text)
            return json_match.group(0) if json_match else None

        def _clean_and_parse_json(json_str: str) -> Union[Dict, str]:
            """Clean and parse JSON string with robust error handling"""
            try:
                # First try direct parse
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If direct parse fails, try cleaning
                try:
                    # Remove markdown markers if present
                    json_str = re.sub(r'```(?:json)?|```', '', json_str).strip()
                    
                    # Fix common JSON issues
                    json_str = json_str.replace("'", '"')  # Single to double quotes
                    json_str = re.sub(r',\s*(}|])', r'\1', json_str)  # Remove trailing commas
                    
                    # Handle unquoted keys
                    json_str = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', json_str)
                    
                    # Try parsing again
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON after cleaning: {e}")
                    return json_str

        def _parse_correct_format(data: Dict) -> Dict:
            """Parse successful answer format"""
            #print(f"data: {data}")
         
            return {
                "Question": question,
                "Answer": final_opinion,
                "Summary": data.get("summary", "") or data.get("Summary", "") 
            }

        def _parse_incorrect_format(data: Dict) -> Dict:
            """Parse incorrect answer format with analysis, handling various key naming conventions"""
            def get_value(keys, default=None):
                for key in keys:
                    if key in data:
                        return data[key]
                return default

            return {
                "Question": question,
                "Correct Answer": official_answer,
                "Initial Hypothesis": get_value([
                    "Initial Hypothesis", 
                    "initialHypothesis", 
                    "InitialHypothesis", 
                    "Initial_Hypothesis", 
                    "initial hypothesis"
                ], ""),
                "Analysis Process": get_value([
                    "Analysis Process",
                    "analysisProcess",
                    "AnalysisProcess",
                    "Analysis_Process",
                    "analysis process"
                ], []),
                "Final Conclusion": final_opinion,
                "Error Reflection": get_value([
                    "Error Reflection",
                    "errorReflection",
                    "ErrorReflection",
                    "Error_Reflection",
                    "error reflection"
                ], "")
            }

        try:
            # Extract JSON
            json_str = _extract_json(response)
            #print(f"Extracted JSON string:\n{json_str}")
            
            if not json_str:
                raise ValueError("No JSON found in response")
                
            # Clean and parse JSON
            parsed_data = _clean_and_parse_json(json_str)
            #print(f"Parsed JSON data:\n{parsed_data}")
            
            if isinstance(parsed_data, str):
                raise ValueError(f"Failed to parse JSON: {parsed_data}")
                
            # Determine format and parse accordingly
            if is_correct:
                result = _parse_correct_format(parsed_data)
            elif not is_correct:
                result = _parse_incorrect_format(parsed_data)
            return result
                
        except Exception as e:
            error_msg = f"Error parsing response: {str(e)}"
            print(f"{error_msg}\nResponse: {response[:200]}...")
            
            return {
                "is_correct": is_correct,
                "error": error_msg,
                "raw_response": response
            }