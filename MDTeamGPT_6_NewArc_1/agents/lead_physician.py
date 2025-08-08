from agents.base_agent import BaseAgent
from typing import List, Dict
import json
import re

class LeadPhysician(BaseAgent):
    def __init__(self):
        super().__init__("Dr. Johnson","Lead Physician")
        self.role_description = """You are a Lead Physician responsible for organizing and summarizing the diagnoses and treatment recommendations provided by specialist doctor agents in each consultation round. Your primary task is to classify all responses from specialists and store them in the Historical Shared Pool for future reference. You do not directly participate in diagnosis but ensure that discussions remain structured, logical, and coherent across multiple rounds."""

    def perform_task(self, 
                     specialist_opinions: List[Dict]
                     ) -> Dict:
        """Analyzes opinions from specialists and returns integrated analysis"""

        prompt = self._build_prompt(
            specialist_opinions
            )
        print("\n====Lead physician Prompt Input====")
        print(prompt)

        response = self._get_llm_response_safe(prompt)
        print("\n====Lead physician Raw Output====")
        print(response)

        parsed_response = self._parse_response(response)  
        print("\n====Lead physician Parsed Output====")
        print(parsed_response)

        return parsed_response            
    

    def _build_prompt(
        self,
        specialist_opinions_summary: str
        ) -> str:

        opinions_text = "\n\n".join(
        f"Specialist: {opinion['Agent_Name']}\n"
        f"Reasoning: {opinion['Reasoning']}\n"
        f"Choice: {opinion['Choice']}"
        for opinion in specialist_opinions_summary
    )

        task_parts = [
            "ROLE DESCRIPTION:",
            f"{self.role_description}",
            "",
            "TASK:",
            "At the end of consultation round {i}, perform these tasks:",
            "1. Collect Responses from All Specialists:",
            "You will receive responses from all specialists in the current consultation round. Each specialist's response follows this format:",
            "{",
            '  "Agent_Name": "Specialist Name",',
            '  "Reasoning": "Explanation of diagnosis and treatment choice",',
            '  "Choice": "{Option ID}: {Option Content}"',
            "}",
            "",
            "2. Categorize and Organize Responses:",
            "Based on the collected responses, systematically classify the information into the following four categories:",
            "• Consistency: Identify diagnoses and treatment recommendations that are similar or identical across multiple specialists.",
            "• Conflict: Highlight contradictory diagnoses or treatment choices among specialists and specify the conflicting points.",
            "• Independence: Extract unique insights mentioned by individual specialists that are not addressed by others.",
            "• Integration: Synthesize a structured summary that incorporates all perspectives, balancing consensus, disagreements, and unique viewpoints.",
            "",
            "3. Store Processed Data in Historical Shared Pool:",
            "Once categorized, save the structured information in the Historical Shared Pool in JSON format:",
            "{",
            '    "consistency": [',
            '      "Summarize common aspects across agents"',
            '    ],',
            '    "conflict": [',
            '      "List conflicting points between agents"',
            '    ],',
            '    "independence": [',
            '      "Extract unique viewpoints from single agents"',
            '    ],',
            '    "integration": [',
            '      "Provide integrated summary of all perspectives"',
            '    ]',
            "}",
            "",
        ]
        processflow_parts = [
            "PROCESS FLOW EXAMPLES:",
            "",
            "1: Collect Responses from Specialist Doctors",
            "Gather responses from all specialists in the current consultation round:",
            "{",
            '  "Obstetrician": {',
            '    "Choice": "{E}: {Nitrofurantoin}",',
            '    "Reasoning": "..."',
            '  },',
            '  "Pathologist": {',
            '    "Choice": "{B}: {Cephalexin}",',
            '    "Reasoning": "..."',
            '  }',
            "}",
            "",
            "2: Categorize Responses",
            "Process all responses into the following categories:",
            "• Consistency: Aggregate similar or identical recommendations.",
            "• Conflict: Identify and document differing opinions among specialists.",
            "• Independence: Extract insights that were mentioned by only one specialist.",
            "• Integration: Formulate a structured summary incorporating all perspectives.",
            "",
            "3: Store Data in Historical Shared Pool",
            "Change categorized content into JSON format:",
            "{",
            '   "consistency": [',
            '     "Summarize the common aspects found across multiple agents."',
            '   ],',
            '   "conflict": [',
            '     "List conflicting points between agents; leave empty if no conflicts exist."',
            '   ],',
            '   "independence": [',
            '     "Extract unique viewpoints mentioned by only one agent; leave empty if none",',
            '     "..."',
            '   ],',
            '   "integration": [',
            '     "Provide a well-structured summary integrating all perspectives"',
            '   ]',
            "}",
            "",
        ]
        specialists_parts=[
            "Specialist_opinions:",
            f"{opinions_text}\n",
        ]
        format_parts=[
            "FORMATTING RULES:",
            "1. Always use double quotes for JSON fields",
            "2. Maintain consistent indentation (2 spaces)",
            "3. Choices must follow {Letter}: {Drug} format",
            "4. Empty arrays should be [] not None/null",
            "5: Just give me the final json format result, do not need to give me anyother text information. "
        ]
        prompt_parts = task_parts+specialists_parts+format_parts
        #prompt_parts = task_parts+processflow_parts+format_parts
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
     
    def _parse_response(self, response: str) -> Dict:
        """Parses LLM response into structured data with 3 fallback methods"""
        # Method 1: Try to find the last complete JSON block
        try:
            # Find all potential JSON blocks
            json_candidates = []
            start_idx = 0
            while True:
                start = response.find('{', start_idx)
                if start == -1:
                    break
                end = response.find('}', start) + 1
                while end <= len(response):
                    try:
                        candidate = response[start:end]
                        # Validate if it's proper JSON
                        parsed = json.loads(candidate)
                        json_candidates.append(parsed)
                        break
                    except json.JSONDecodeError:
                        # Look for next closing brace
                        end = response.find('}', end) + 1
                        if end == 0:  # No more closing braces
                            break
                start_idx = end if end > start_idx else start_idx + 1
            
            if json_candidates:
                # Return the most complete JSON structure (prioritize those with round data)
                for candidate in reversed(json_candidates):
                    if 'round' in str(candidate):
                        return candidate
                return json_candidates[-1]
        except Exception:
            pass
            
        # Method 3: Return error if all parsing fails
        return {
            "error": "Failed to parse LLM response",
            "raw_response": response
        }