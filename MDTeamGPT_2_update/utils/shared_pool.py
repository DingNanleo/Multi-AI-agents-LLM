import os
import json
from typing import Dict, List
from datetime import datetime


class HistoricalSharedPool:
    def __init__(self,case_id=None):
        self.pool = {
            "question_id": case_id,
            "question": None,
            "options": None,
            "timestamp": datetime.now().isoformat(),
            "primary_care_assignment": None
            # Note: rounds will be added as "round 1", "round 2" etc. at top level
        }
        self.current_round = 0  # Tracks the current round number
        self.case_id = case_id
        self.save_directory = "historical_pool"
        self.filename = None  # Will be set when first saving
        os.makedirs(self.save_directory, exist_ok=True)
        if self.case_id:
            self._set_filename()

    def _set_filename(self):
        """Internal method to set the filename once"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #print(f"\ncase_id: {self.case_id}")
        if self.case_id:
            base_name = f"case_{self.case_id}_{timestamp}"
        else:
            base_name = f"case_unknown_{timestamp}"
        
        # Ensure filename is unique by adding counter if needed
        counter = 1
        self.filename = f"{self.save_directory}/{base_name}.json"
        while os.path.exists(self.filename):
            self.filename = f"{self.save_directory}/{base_name}_{counter}.json"
            counter += 1
           
    def add_statements(self, statements: Dict):
        """Store statements and append to existing file"""
        if not statements:
            return
        for key, value in statements.items():
            if key in self.pool:
                # If both values are dictionaries, merge them
                if isinstance(self.pool[key], dict) and isinstance(value, dict):
                    self.pool[key].update(value)
                # If both values are lists, extend them
                elif isinstance(self.pool[key], list) and isinstance(value, list):
                    self.pool[key].extend(value)
                else:
                    # Otherwise, overwrite the existing value
                    self.pool[key] = value
            else:
                # New key, just add it
                self.pool[key] = value

        print(f"\nstatements: {statements}")
        print(f"\nStored statements: {list(statements.keys())}")
        self.save_to_file()

    def save_to_file(self):
        """Save the current state of the pool to the same file"""
        if self.filename is None:
            self._set_filename()  # Ensure filename is set
        
        # Write the complete current state of the pool
        with open(self.filename, 'w') as f:
            json.dump(self.pool, f, indent=2)
            
        print(f"Updated historical pool at {self.filename}")


    def get_all_statements(self) -> Dict:
        #print("share pool: get_all_statements")
        all_data = {
            "metadata": self._get_metadata()
            #"round": self._get_rounds_data()
        }
        #print(f"share pool: get_all_statements data:{all_data}")
        return all_data

    def _get_metadata(self) -> Dict:
        """Get all non-round specific metadata"""
        return {
            k: v for k, v in self.pool.items()
            if not (isinstance(k, int)) or 
                (isinstance(k, str) and k.startswith("round "))
        }

    def _get_rounds_data(self) -> Dict[int, Dict]:
        """Get all available round data"""
        rounds = {}
        
        for key in self.pool.keys():
            if isinstance(key, int):
                rounds[key] = self.get_round_statements(key)
            elif isinstance(key, str) and key.startswith("round "):
                try:
                    round_num = int(key.split()[1])
                    rounds[round_num] = self.get_round_statements(round_num)
                except (IndexError, ValueError):
                    continue
        
        return rounds

    def get_round_statements(self, round_num: int) -> Dict:
        """Get data for a specific round"""
        round_key = f"round {round_num}"
        round_data = self.pool.get(round_num, self.pool.get(round_key, {}))
        
        return {
            "specialist_opinions": round_data.get("specialist_opinions", []),
            "lead_physician_analysis": round_data.get("Lead_Physician_Opinion", {})
        }


    def get_lead_physician_opinions(self, last_n_rounds: int = 1) -> List[Dict]:

        #all_rounds = self.get_all_statements()
        all_rounds = self._get_rounds_data()
        sorted_rounds = sorted(all_rounds.keys(), reverse=True)
        
        results = []
        #print(f"share pool: get_lead_physician_opinions_all_rounds:{all_rounds}")
        for round_num in sorted_rounds[:last_n_rounds]:
            round_data = all_rounds[round_num]
            if round_data["lead_physician_analysis"]:
                results.append(round_data["lead_physician_analysis"])

        #print(f"result of get lead physician: {results}")

        return results

    def get_specialist_opinions(self) -> List[Dict]:
        """Get all specialist opinions from all historical rounds"""
        all_rounds = self.get_all_statements()
        
        results = []
        for round_data in all_rounds.values():
            if round_data["specialist_opinions"]:
                results.extend(round_data["specialist_opinions"])
        
        return results

    def get_specialist_opinions_by_round(self, round_num: int) -> List[Dict]:
        round_data = self.get_round_statements(round_num)
        return round_data["specialist_opinions"]

    def clear_pool(self):
        """Clear all statements from the pool"""
        self.pool = {}
        self.current_round = 0

#----------------------Additional utility methods----------------------
    def get_specialist_opinions(self, round_num: int) -> List[Dict]:
        """Get only specialist opinions for a specific round"""
        return self.get_round_statements(round_num).get("specialist_opinions", [])

    def get_lead_analysis(self, round_num: int) -> Dict:
        """Get only lead physician analysis for a specific round"""
        return self.get_round_statements(round_num).get("lead_physician_analysis", {})
    
    def get_lastone_statements(self) -> Dict:
        """Return all statements in the pool"""
        return self.pool

    def get_lasttwo_statements(self)-> Dict:
        """Return all statements in the pool"""
        return self.pool

class CorrectAnswerKnowledgeBase:
    def __init__(self):
        self.knowledge_base = []
        
    def add_case(self, patient_background: str, medical_problem: str, final_opinion: str, metadata: dict = None):
        entry = {
            "patient_background": patient_background,
            "medical_problem": medical_problem,
            "final_opinion": final_opinion,
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.knowledge_base.append(entry)
        
    def search_similar_cases(self, medical_problem: str, threshold: float = 0.7) -> List[dict]:
        similar = []
        for case in self.knowledge_base:
            if medical_problem.lower() in case["medical_problem"].lower():
                similar.append(case)
        return similar