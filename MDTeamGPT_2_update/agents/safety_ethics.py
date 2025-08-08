import json
import re
from typing import Dict,Optional,List,Union
from agents.base_agent import BaseAgent

class SafetyEthicsReviewer(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Ethics", "Safety & Ethics Reviewer")
        
    def perform_task(self, 
                     final_opinion: str, 
                     question: str, 
                     options:str,
                     history: Dict) -> Dict:
     
        prompt = [ 
            f"Role: You are {self.role}",
            f"Question: {question}",
            f"Options: {options}",
            f"Agent's answer: {final_opinion}",

            "Review the agent diagnosis for safety risks, ethical concerns, and biases, ensuring no harmful or unethical recommendations are present. Refine the final advice to align with medical ethics and patient safety standards.",
            "Return ONLY valid JSON in this EXACT format (no extra text or explanations):",
            '```json',
            '{',
            '   "approval: boolean",',   
            '   "ethics_answer: {Option ID}: {Option Content}",',
            '   "ethics_analysis: analysis of safety risks, ethical issues, potential biases if having, string (max 3 sentences)",',
            '}',
            '```',
            "Example:",
            '```json',
            '{',
            '   "approval: true",',   
            '    "ethics_answer: {E}: {Nitrofurantoin}",',
            '    "ethics_analysis: ......."',
            '}',
            '```'
        ]

        
        full_prompt = "\n".join(prompt)
        print("\n=====safety ethics prompt=====")
        print(full_prompt)

        response = self.call_llm(full_prompt, max_tokens=500)
        print("\n=====safety ethics raw output=====")
        print(response)

        print("\n=====safety ethics parsed output=====")
        parsed_response = self._parse_response(response)
        print(parsed_response)

        return parsed_response


    def _parse_response(self, response: str) -> Dict:
        # Try parsing as JSON first
        try:
            parsed = json.loads(response)
            return {
                "approval": parsed.get("approval", False),
                "ethics_answer": parsed.get("ethics_answer"),
                "ethics_analysis": parsed.get("ethics_analysis")
            }
        except json.JSONDecodeError:
            pass  # Fall back to manual parsing

        # Fallback parsing for malformed JSON
        approval = None
        ethics_answer = None
        ethics_analysis = None

        # Regex to match key: value pairs (even if unquoted)
        pattern = re.compile(r'^\s*"?(approval|ethics_answer|ethics_analysis)"?\s*:\s*(.+)$', re.IGNORECASE)

        for line in response.splitlines():
            match = pattern.match(line.strip())
            if match:
                key = match.group(1).lower()
                value = match.group(2).strip().strip('",')
                if key == 'approval':
                    approval = value.lower() == 'true'
                elif key == 'ethics_answer':
                    ethics_answer = value
                elif key == 'ethics_analysis':
                    ethics_analysis = value

        return {
            "approval": approval,
            "ethics_answer": ethics_answer,
            "ethics_analysis": ethics_analysis
        }



    # def _parse_response(self, response: str) -> Dict:
    #     # Initialize variables
    #     approval = None
    #     ethics_answer = None
    #     ethics_analysis = None
        
    #     # Split the response into lines
    #     lines = response.split('\n')
        
    #     for line in lines:
    #         line = line.strip()
    #         if line.startswith('"approval:'):
    #             # Extract approval value
    #             approval = line.split(':', 1)[1].strip().strip('",')
    #             approval = approval.lower() == 'true'
    #         elif line.startswith('"ethics_answer:'):
    #             # Extract ethics_answer value
    #             ethics_answer = line.split(':', 1)[1].strip().strip('",')
    #         elif line.startswith('"ethics_analysis:'):
    #             # Extract ethics_analysis value
    #             ethics_analysis = line.split(':', 1)[1].strip().strip('",')
        
    #     return {
    #         "approval": approval,
    #         "ethics_answer": ethics_answer,
    #         "ethics_analysis": ethics_analysis
    #     }