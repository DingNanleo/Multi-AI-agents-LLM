import json
from typing import Dict, List, Any
from utils.vector_db import VectorDatabase

class ChainOfThoughtReviewer:
    def __init__(self):
        self.correct_kb = VectorDatabase("correct_answers")
        self.chain_kb = VectorDatabase("chain_of_thought")
    
    def perform_task(
            self, 
            final_opinion: str,
            official_answer: str, 
            patient_background: str,
            medical_problem_with_options: str,  # Changed from Dict to str
            historical_pool: Dict,  # Fixed capitalization
            metadata: Dict) -> Dict:
        """
        Process the chain of thought and store in appropriate knowledge base
        Returns: {
            "status": "processed",
            "correctness": bool,
            "storage_location": str
        }
        """
        chain_data = self._extract_chain_of_thought(historical_pool, official_answer)
        
        self._store_chain_of_thought(
            chain_data,
            {
                "medical_problem": medical_problem_with_options,
                "patient_background": patient_background,
                "official_answer": official_answer,
                "final_opinion": final_opinion,
                **metadata  # Unpack additional metadata
            }
        )
        
        return {
            "status": "processed",
            "correctness": chain_data["is_correct"],
            "storage_location": "correct_kb" if chain_data["is_correct"] else "chain_kb"
        }
        
    def _extract_chain_of_thought(self, historical_pool: dict, correct_answer: str) -> dict:
        """
        Process historical pool to extract structured chain of thought
        Handles the nested list structure in the historical pool
        """
        result = {
            "initial_hypothesis": [],
            "analysis_process": [],
            "final_conclusion": "",
            "is_correct": False
        }

        #print("\nDEBUG - Historical pool structure:")
        #print(json.dumps(historical_pool, indent=2))
        
        for round_key, round_data in historical_pool.items():
            try:
                round_num = int(round_key.split()[-1])  # Extract round number
                
                # Handle specialist opinions (first element in the list)
                if isinstance(round_data, list) and len(round_data) > 0:
                    specialist_opinions = round_data[0]  # This is the list of specialist opinions
                    
                    if isinstance(specialist_opinions, list):
                        for opinion in specialist_opinions:
                            if isinstance(opinion, dict):
                                record = {
                                    "agent": opinion.get("Agent_Name", "Unknown"),
                                    "reasoning": opinion.get("Reasoning", ""),
                                    "choice": opinion.get("Choice", ""),
                                    "round": round_num
                                }
                                
                                if round_num == 1:
                                    result["initial_hypothesis"].append(record)
                                else:
                                    record["conflict"] = False
                                    result["analysis_process"].append(record)
                
                # Handle lead physician analysis (second element in the list)
                if isinstance(round_data, list) and len(round_data) > 1:
                    lead_analysis = round_data[1]  # This is the lead physician analysis
                    
                    if isinstance(lead_analysis, dict) and lead_analysis.get("Agent_Name") == "Lead Physician":
                        result["final_conclusion"] = ", ".join(lead_analysis.get("integration", []))
                        
                        # Mark conflicts in analysis process
                        for conflict in lead_analysis.get("conflict", []):
                            for item in result["analysis_process"]:
                                if conflict.lower() in item.get("reasoning", "").lower():
                                    item["conflict"] = True
            
            except (AttributeError, ValueError, IndexError) as e:
                print(f"Warning: Could not process round {round_key}: {str(e)}")
                continue
        
        # Determine correctness
        result["is_correct"] = any(
            correct_answer.lower() in item.get("choice", "").lower()
            for item in result["initial_hypothesis"] + result["analysis_process"]
        )
        
        # Add error analysis if incorrect
        if not result["is_correct"]:
            result["error_analysis"] = self._analyze_errors(
                result["analysis_process"],
                correct_answer
            )
        
        return result

    def _analyze_errors(self, analysis_process: list, correct_answer: str) -> dict:
        """Identify where reasoning went wrong"""
        error_points = []
        for step in analysis_process:
            if step["conflict"]:
                error_points.append({
                    "round": step["round"],
                    "agent": step["agent"],
                    "incorrect_reasoning": step["reasoning"],
                    "likely_error": "Conflict with other specialists"
                })
        
        return {
            "error_points": error_points,
            "correct_answer": correct_answer,
            "suggested_improvements": [
                "Consider alternative diagnoses when conflicts arise",
                "Verify initial assumptions with clinical guidelines"
            ]
        }

    def _store_chain_of_thought(self, chain_data: dict, metadata: dict):
        """Store in appropriate knowledge base"""
        if chain_data["is_correct"]:
            self._store_correct_chain(chain_data, metadata)
        else:
            self._store_error_chain(chain_data, metadata)

    def _store_correct_chain(self, chain: dict, metadata: dict):
        """Store in CorrectKB"""
        record = {
            "Background": metadata["patient_background"],
            "Question": metadata["medical_problem"],
            "Answer": self._extract_final_choice(chain),
            "Reasoning_Chain": chain["analysis_process"],
            "Consensus_Points": chain["final_conclusion"],
            "metadata": {
                "patient_background": metadata["patient_background"],
                "specialists": metadata.get("specialists", []),
                "rounds": len({item['round'] for item in chain["analysis_process"]})
            }
        }
        self.correct_kb.store(record)

    def _store_error_chain(self, chain: dict, metadata: dict):
        """Store in ChainKB"""
        record = {
            "Background": metadata["patient_background"],
            "Question": metadata["medical_problem"],
            "Correct_Answer": metadata["official_answer"],
            "Initial_Hypothesis": chain["initial_hypothesis"],
            "Analysis_Process": chain["analysis_process"],
            "Final_Conclusion": metadata["final_opinion"],
            "Error_Analysis": chain.get("error_analysis", {}),
            "metadata": {
                "patient_background": metadata["patient_background"],
                "specialists": metadata.get("specialists", []),
                "rounds": len({item['round'] for item in chain["analysis_process"]})
            }
        }
        self.chain_kb.store(record)

    def _extract_final_choice(self, chain: dict) -> str:
        """Extract most recent choice from analysis process"""
        if not chain["analysis_process"]:
            return chain["initial_hypothesis"][0]["choice"] if chain["initial_hypothesis"] else ""
        return chain["analysis_process"][-1]["choice"]