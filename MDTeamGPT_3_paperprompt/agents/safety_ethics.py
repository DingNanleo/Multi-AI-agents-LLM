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
        """
        Enhanced safety and ethics review with four-step process
        Args:
            options: Dictionary of {option_id: option_content} for medication/procedure choices
        """
        prompt = self._build_review_prompt(final_opinion, question, options, history)
        
        print("\n=====safety ethics prompt=====")
        print(prompt)

        llm_response = self.call_llm(prompt)
        print("\n=====safety ethics raw output=====")
        print(llm_response)

        parse_response = self._parse_response(llm_response,final_opinion)
        print("\n=====safety ethics parsed output=====")
        print(parse_response)

        return parse_response
    
    def _build_review_prompt(self, final_opinion: str, question: str, options:str,
                           historical_pool: Dict) -> str:
        base_parts = [
            "ROLE DESCRIPTION:",
            "You are a Safety and Ethics Reviewer, responsible for ensuring that the final diagnosis and treatment recommendation are free from harmful information. Your task is to review the doctors' final opinions, filter out any harmful or unethical suggestions, and output a safe conclusion along with the corresponding answer ID.",
            "",
            "Task:",
            "1. Review Final Opinions:",
            "Review the doctor's final diagnosis and advice to ensure that there are no unethical or harmful information.",
            "2. Filter Harmful Information:",        
            "In the final diagnosis, identify any suggestions that are harmful, unethical or unsafe. This may include unsafe medications, incorrect procedures, or advice that could negatively impact a patient's well-being or safety.",
            "3. Refine and Confirm Conclusion:",
            "Refine the diagnosis after filtering out any harmful information to ensure that it is safe and ethical. Verify the final treatment recommendation to ensure that it is in line with the highest standards of medical ethics.",
            "4. Output Conclusion and Answer:",
            "Provide a clean and accurate conclusion by outputting both the answer ID and recommended treatment option as an output in this format:",
            "Answer ID: {Option ID}: {Option Content}.",
            "",
            "Output format(do not use ** in the answer's titles, follow this output format strictly):",
            "1.Review Final Opinion:.......",
            "2.Filter Harmful Information:........",
            "3.Refine and Confirm Conclusion:........",
            "4.Output Conclusion and Answer:",
            "{Option ID}: {Option Content}",
            "for example:{C}: {6th aortic arch}",
            "",
        ]

        patient_parts = [
            "PATIENT CASE:",
            f"Background: {question}",
            f"Options: {options}",
            "",
            f"Specialist_Opinion: {final_opinion}"
            #f"History: {historical_pool}",
            ""
        ]

        example_part = [
            "EXAMPLE:",
            "Patient Description: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract.",
            "She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air.",
            "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
            "",
            "Safety Review Example:",
            "1. Review Final Opinions:",
            "The doctors suggested several antibiotics for treating the urinary tract infection, including Ciprofloxacin and Nitrofurantoin.",
            "2. Filter Harmful Information:",
            "Ciprofloxacin is not recommended for pregnant women due to potential harm to the fetus. It was identified as unsafe and filtered out from the final recommendation.",
            "3. Refine and Confirm Conclusion:",
            "After filtering out Ciprofloxacin, Nitrofurantoin was confirmed as the safest option for treating the patient's condition during pregnancy.",
            "4. Output Conclusion and Answer:",
            "Answer ID: {E}: {Nitrofurantoin}",
            ""
        ]

        #full_prompt = "\n".join(base_parts + patient_parts + example_part)

        full_prompt = "\n".join(base_parts + patient_parts)

        return full_prompt
    
    def _parse_response(self, llm_response: str,final_opinion:str) -> Dict:

        harmful_filter_match = re.search(
        r"2\.\s*Filter Harmful Information:\s*(.+?)(?=\n3\.|$)",
        llm_response,
        re.DOTALL | re.IGNORECASE
        )
        harmful_filter = harmful_filter_match.group(1).strip() if harmful_filter_match else "No harmful information filter provided."

        # Extract the refined conclusion (step 3)
        conclusion_match = re.search(
            r"3\.\s*Refine and Confirm Conclusion:\s*(.+?)(?=\n4\.|$)", 
            llm_response, 
            re.DOTALL | re.IGNORECASE
        )
        conclusion = conclusion_match.group(1).strip() if conclusion_match else "No conclusion provided."

        combined_conclusion = f"Filter Harmful Information: {harmful_filter}\nRefined Conclusion: {conclusion}"

        # clean_response = re.sub(r'\*\*|\*|__', '', llm_response)  # Remove markdown
        # clean_response = re.sub(r'\n+', '\n', clean_response.strip())  # Normalize newlines

        answer_patterns = [
            # Primary pattern - handles numbered output with braces
            r"4\.\s*Output Conclusion and Answer:\s*\{([A-Z])\}:\s*\{([^}]+)\}",
            # Handles unnumbered version
            r"Output Conclusion and Answer:\s*\{([A-Z])\}:\s*\{([^}]+)\}",
            # Fallback for unbraced answers
            r"Output Conclusion and Answer:\s*([A-Z]):\s*([^\n]+)",
            # Final fallback - any option pattern
            r"([A-Z])\s*:\s*([^\n]+)"
        ]
        
        for pattern in answer_patterns:
            answer_match = re.search(
                pattern,
                llm_response,
                re.DOTALL | re.IGNORECASE
            )
            #print(f"\n answer_match:{answer_match} \n")
            if answer_match:
                option = answer_match.group(1).strip()
                answer_text = answer_match.group(2).strip()
                answer = f"{{{option}}}: {answer_text}"  # Wrap option in curly braces
            else:
                answer = "No answer provided."

            return {
                "Agent_Name":"Safety Ethics",
                "Specialists Answer": final_opinion,
                "Ethics Answer": answer,
                "Ethics Conclusion": combined_conclusion
        }
