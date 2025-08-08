import json
import re
from typing import Dict
from agents.base_agent import BaseAgent

class SafetyEthicsReviewer(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Ethics","Safety & Ethics Reviewer")
        self.role_description = """You are a Safety and Ethics Reviewer, responsible for ensuring that the final diagnosis and treatment
recommendation are free from harmful information. Your task is to review the doctors’ final opinions, filter
out any harmful or unethical suggestions, and output a safe conclusion along with the corresponding answer
ID."""
        
    def perform_task(self, 
                    final_opinion: str, 
                    question: str, 
                    options: str,
                    history: Dict) -> Dict:
        
        prompt_part = "\n".join([
            f"Role: {self.name} ({self.role})",
            f"Question: {question}",
            f"Options: {options}",
            f"Agent's Diagnosis: {final_opinion}",
            "",
            "Task:",
            "1. Review Agent's Diagnosis:",
            "Review the doctor's final diagnosis to ensure that there are no unethical or harmful information.",
            "2. Filter Harmful Information:",        
            "In the final diagnosis, identify any suggestions that are harmful, unethical or unsafe. This may include unsafe medications, incorrect procedures, or advice that could negatively impact a patient's well-being or safety.",
            "3. Refine and Confirm Conclusion:",
            "Refine the diagnosis after filtering out any harmful information to ensure that it is safe and ethical. Verify the final treatment recommendation to ensure that it is in line with the highest standards of medical ethics.",
            "4. Output Conclusion and Answer:",
            "Provide a clean and accurate conclusion in this exact format (copy this format exactly):",
            "{",
            "approval: boolean",
            "recommended_answer: {Option ID}: {Option Content}",
            "conclusion: string (max 3 sentences)",
            "}",
            "Example:",
            "{",
            "approval: true",
            "recommended_answer: {E}: {Nitrofurantoin}",
            "conclusion: The patient is at high risk for anaphylaxis. No ethical concerns. No apparent biases.",
            "}"
        ])

        try:
            print("\n=====safety ethics prompt=====")
            print(prompt_part)

            print("\n=====safety ethics raw output=====")
            llm_response = self.call_llm(prompt_part)
            print(llm_response)

            print("\n=====safety ethics parsed output=====")
            parse_response = self._parse_response(llm_response)
            print(parse_response)

            return parse_response
            
        except Exception as e:
            print(f"API Error: {str(e)}")
            return {
                "approval": False,
                "recommended_answer": "Error: Invalid response",
                "conclusion": "Failed to process ethics review"
            }
    
    def _parse_response(self, response: str) -> Dict:
        """
        Robust response parser that handles both JSON and loose formats
        """
        try:
            # Clean the response first
            cleaned = response.strip().replace('\n', ' ').replace('\t', ' ')
            
            # Try to parse as JSON first (if properly formatted)
            try:
                result = json.loads(cleaned)
                return self._validate_structure(result)
            except json.JSONDecodeError:
                pass
            
            # Fallback to parsing loose format
            parsed = {}
            
            # Extract approval
            approval_match = re.search(r'approval:\s*(true|false)', cleaned, re.IGNORECASE)
            if approval_match:
                parsed["approval"] = approval_match.group(1).lower() == 'true'
            
            # Extract recommended answer
            answer_match = re.search(r'recommended_answer:\s*\{([A-Z])\}:\s*\{(.*?)\}', cleaned)
            if answer_match:
                parsed["recommended_answer"] = {answer_match.group(1): answer_match.group(2)}
            
            # Extract conclusion
            conclusion_match = re.search(r'conclusion:\s*(.*?)(?:\}$|$)', cleaned)
            if conclusion_match:
                parsed["conclusion"] = conclusion_match.group(1).strip()
            
            return self._validate_structure(parsed)
            
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}")

    def _validate_structure(self, data: Dict) -> Dict:
        """Validates and standardizes the response structure"""
        if not all(k in data for k in ["approval", "recommended_answer", "conclusion"]):
            raise ValueError("Missing required fields in response")
        
        # Ensure recommended_answer is in the correct format
        if isinstance(data["recommended_answer"], str):
            match = re.match(r'\{([A-Z])\}:\s*\{(.*?)\}', data["recommended_answer"])
            if match:
                data["recommended_answer"] = {match.group(1): match.group(2)}
            else:
                raise ValueError("Invalid recommended_answer format")
        
        return {
            "approval": bool(data["approval"]),
            "recommended_answer": data["recommended_answer"],
            "conclusion": str(data["conclusion"]).strip()
        }




    # def _parse_response(self, response: str) -> Dict:
    #     """
    #     Simplified response parser that extracts the JSON structure
    #     """
    #     try:
    #         # Extract the first JSON object found in the response
    #         json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
    #         result = json.loads(json_str)
            
    #         # Basic structure validation
    #         if not all(k in result for k in ["approval", 
    #                                        "recommended_answer", 
    #                                        "conclusion"]):
    #             raise ValueError("Missing required fields")
            
    #         # Convert recommended answer to dict if it's not already
    #         if isinstance(result["recommended_answer"], str):
    #             # Handle cases where LLM might output string like "{E}: {Nitrofurantoin}"
    #             match = re.match(r'\{([A-Z])\}:\s*\{(.*?)\}', result["recommended_answer"])
    #             if match:
    #                 result["recommended_answer"] = {match.group(1): match.group(2)}
    #             else:
    #                 raise ValueError("Couldn't parse recommended answer format")
            
    #         return {
    #             "approval": bool(result["approval"]),
    #             "recommended_answer": result["recommended_answer"],
    #             "conclusion": str(result["conclusion"])
    #         }
            
    #     except Exception as e:
    #         raise ValueError(f"Failed to parse response: {str(e)}")
