import datetime
from typing import Dict, List, Any

class HistoricalSharedPool:
    def __init__(self):
        self.pool = {}  # Stores all historical statements
        self.current_round = 0  # Tracks the current round number

    def add_statements(self, statements: Dict):
        """
        Store statements in the pool with automatic round tracking
        Expected format: {"round X": [list of statements]} or {"round X": {statement dict}}
        """
        if not statements:
            return
            
        # Extract round number from the key
        round_key = next(iter(statements.keys()))  # Get the first key
        if not round_key.startswith("round "):
            raise ValueError("Statement keys must be in format 'round X'")
            
        # Update current round
        try:
            round_num = int(round_key.split(" ")[1])
            self.current_round = max(self.current_round, round_num)
        except (IndexError, ValueError):
            raise ValueError("Invalid round number format in key")
        
        # Add to pool, merging if round already exists
        if round_key in self.pool:
            # If existing value is a list and new value is a list, extend
            if isinstance(self.pool[round_key], list) and isinstance(statements[round_key], list):
                self.pool[round_key].extend(statements[round_key])
            # If existing is a dict and new is a dict, merge them
            elif isinstance(self.pool[round_key], dict) and isinstance(statements[round_key], dict):
                self.pool[round_key].update(statements[round_key])
            else:
                # Different types - store as list containing both
                self.pool[round_key] = [self.pool[round_key], statements[round_key]]
        else:
            # New round - just store the statements
            self.pool[round_key] = statements[round_key]
            
        print(f"Stored statements for {round_key}")

    def get_all_statements(self) -> Dict:
        """Return all statements in the pool"""
        return self.pool

    def get_round_statements(self, round_num: int):
        """Get statements for a specific round"""
        round_key = f"round {round_num}"
        return self.pool.get(round_key, None)

    def clear_pool(self):
        """Clear all statements from the pool"""
        self.pool = {}
        self.current_round = 0

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