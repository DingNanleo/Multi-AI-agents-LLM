from agents.base_agent import BaseAgent
from typing import Dict, List, Optional, TYPE_CHECKING
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

if TYPE_CHECKING:
    from utils.shared_pool import HistoricalSharedPool

class SpecialistDoctor(BaseAgent):
    def __init__(self, specialty: str):
        super().__init__(f"Dr. {specialty}", f"{specialty.capitalize()} Specialist")
        self.specialty = specialty
        #self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
        self.correctKB = []  # Stores confirmed correct cases
        self.chainKB = []    # Stores debated cases
        
    def perform_task(
            self, 
            patient_background: str, 
            medical_problem: str, 
            options: Optional[List[str]],
            round_num: int, 
            historical_pool: Optional['HistoricalSharedPool'] = None,
            correctKB: Optional[List] = None,
            chainKB: Optional[List] = None
    ) -> Dict:
        
        """Enhanced opinion generation for MedQA format with structured rounds"""
        try:
            # Validate inputs
            if not all([patient_background, medical_problem]):
                raise ValueError("Missing required patient information")
            # Update KBs if provided
            if correctKB is not None:
                self.correctKB = correctKB
            if chainKB is not None:
                self.chainKB = chainKB
            #print(f"\n correctKB:{correctKB}")
            #print(f"\n chainKB:{chainKB}")

            # Get history if available
            history = []
            if historical_pool and round_num > 1:
                history = self._get_relevant_history_safe(round_num, historical_pool)
                #print(f"\n history:{history}")

            # Build MedQA-specific prompt with round structure
            prompt = self._build_prompt(
                background=patient_background,
                question=medical_problem,
                options=options,
                round_num=round_num,
                history=history
            )
            # Print prompt for rounds > 1
            #if round_num > 1:
                #print("\n=== SPECIALIST PROMPT (ROUND {}) ===".format(round_num))
                #print(prompt)
            
            # Get LLM response
            response = self._get_llm_response_safe(prompt)

            # Print response for rounds > 1
            #if round_num > 1:
                #print("\n=== SPECIALIST RESPONSE (ROUND {}) ===".format(round_num))
                #print(response)
            
            # Format response for MedQA
            return self._format_response(
                response=response
            )
            
        except Exception as e:
            print(f"Error in perform_task: {str(e)}")
            return self._create_error_response(round_num)
        
    def _retrieve_similar_cases(self, background: str, problem: str, k: int = 3) -> List[Dict]:
        """TF-IDF based similarity search"""
        query = f"{background} {problem}"
        #print("\n===Retrieve similar case===")
        # Prepare corpus
        corpus = []
        case_references = []
        for kb in [self.correctKB, self.chainKB]:
            for case in kb:
                case_text = f"{case.get('metadata', {}).get('patient_background', '')} {case.get('Question', '')}"
                corpus.append(case_text)
                case_references.append(case)
        
        if not corpus:
            return []

        # Vectorize
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            X = vectorizer.fit_transform(corpus + [query])
            similarities = cosine_similarity(X[-1], X[:-1])[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            # Format the simplified output
            simplified_cases = []
            for i in top_indices:
                case = case_references[i]
                simplified_cases.append({
                    'background': case.get('metadata', {}).get('patient_background', ''),
                    'question_with_options': case.get('Question', ''),
                    'correct_answer': case.get('Correct_Answer', '') or case.get('Answer', ''),
                    'source': 'CorrectKB' if i < len(self.correctKB) else 'ChainKB',
                    'similarity': float(similarities[i])
                })
            
            return simplified_cases
    
        except ValueError:
            return []

    def _build_prompt(
            self,
            background: str,
            question: str,
            options: List[str],
            round_num: int,
            history: List[Dict]) -> str:
        """Constructs a structured prompt following MedQA round examples"""
        prompt_parts = [
            f"ROLE: You are {self.role}, a specialist in {self.specialty}",
            f"TASK: Analyze this medical case and select the best option (Round {round_num})",
            "\nPATIENT BACKGROUND:", background[:1000],  # Increased length for clinical details
            "\nQUESTION:", question,
            "\nOPTIONS:"
        ]
        
        # Add formatted options
        option_letters = [chr(65 + i) for i in range(len(options))]
        for letter, option in zip(option_letters, options):
            prompt_parts.append(f"{letter}: {option}")
        
        # Add KB context for rounds > 1
        if round_num > 1:
            similar_cases = self._retrieve_similar_cases(background, question)
            #print(f"\n similarcase:{similar_cases}")

            if similar_cases:
                kb_context = ["\nRELEVANT PRIOR CASES:"]
                for i, case in enumerate(similar_cases):
                    prompt_parts.append(f"\n=== Case {i+1} [{case['source']}] ===")
                    prompt_parts.append(f"Background: {case['background']}")
                    prompt_parts.append(f"Question: {case['question_with_options']}")
                    prompt_parts.append(f"Correct Answer: {case['correct_answer']}")
                    prompt_parts.append(f"Similarity Score: {case['similarity']:.2f}")
        
        # Add round-specific instructions
        prompt_parts.append(f"\nROUND {round_num} ANALYSIS STRUCTURE:")
        if round_num == 1:
            prompt_parts.extend([
                "1. Patient Condition Analysis: [Carefully read the patient’s description of symptoms, combining their signs, clinical examination, and pregnancy status for a comprehensive analysis]",
                "2. Treatment Option Evaluation: [Based on your professional knowledge, analyze all available treatment options, paying particular attention to drug safety for both the pregnant woman and the fetus]",
                "3. Select Optimal Treatment Plan: [Determine the most appropriate treatment for the patient and explain your decision]",
                "4. Express Conclusion: [Use the following format to express your conclusion: Choice: {Option ID}: {Option Content} Example: Choice: {E}: {Nitrofurantoin}]"
            ])
        else:
            prompt_parts.extend([
                "1. Patient Condition Analysis: [Carefully read the patient’s description of symptoms, combining their signs, clinical examination, and pregnancy status for a comprehensive analysis]",
                "2. Treatment Option Evaluation: [Based on your professional knowledge, analyze all available treatment options, paying particular attention to drug safety for both the pregnant woman and the fetus]",
                f"3. Review Round {round_num-1} Feedback: [Reference prior discussions]",
                "4. Select Optimal Treatment Plan: [Determine the most appropriate treatment for the patient and explain your decision]",
                "5. Express Conclusion: [Use the following format to express your conclusion: Choice: {Option ID}: {Option Content} Example: Choice: {E}: {Nitrofurantoin}]"
            ])
       
        if history:
            prompt_parts.append("\nLEAD PHYSICIAN ANALYSIS FROM PREVIOUS ROUNDS:")
            for item in history:
                # Format independence as "Specialist: statement1; statement2"
                independence_data = item.get('independence', {})
                formatted_independence = []
                for specialist, statements in independence_data.items():
                    formatted_statements = "; ".join(statements)
                    formatted_independence.append(f"{specialist}: {formatted_statements}")
                
                analysis = [
                    f"- Round {item['round']} Summary:",
                    f"  Consistency: {', '.join(item.get('consistency', [])) or 'None'}",
                    f"  Conflict: {', '.join(item.get('conflict', [])) or 'None'}",
                    f"  Independence: {', '.join(formatted_independence) or 'None'}",  # Updated line
                    f"  Integration: {' | '.join(item.get('integration', [])) or 'None'}"
                ]
                prompt_parts.extend(analysis)
        else:
            prompt_parts.append("\nLEAD PHYSICIAN ANALYSIS: [No prior analysis available]")

        # Add response format requirements
        prompt_parts.extend([
            "\nRESPONSE REQUIREMENTS:",
            "- Must select exactly one option",
            "- Must follow the specified analysis structure",
            "- Must give answers in text without bullets ",
            "- Maintain spaces after numbers (1. ) and colons (: )",
            "- Use less then 3 sentences to show answer"
        ])
        
        full_prompt = "\n".join(prompt_parts)
        
        # Debug print
        #if round_num > 1:
            #print("\n===Specialist Prompt Input===")
            #print(full_prompt)
        
        return full_prompt


    def _get_llm_response_safe(self, prompt: str, max_retries: int = 3) -> str:
        """Safe LLM communication with retries"""
        for attempt in range(max_retries):
            try:
                response = self.call_llm(prompt, max_tokens=1000)  # Increased tokens
                if response and len(response) > 20:
                    print(f"\n===Specialist (Raw) Output Attempt {attempt + 1}:===")
                    print(response)
                    return response
                print(f"Attempt {attempt+1}/{max_retries}: Empty or short response")
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
        raise ValueError("Failed to get valid LLM response")


    def _format_response(self, 
                        response: str) -> Dict:
        """Formats and validates the LLM response according to MedQA standards"""
        try:
            # Extract structured components
            reason = self._parse_section(response, "Select Optimal Treatment Plan")
            choice = self._parse_section(response, "Express Conclusion")
            #print(f"\n Choice:{choice}")
            return {
                "Agent_Name": self.role,
                "Reasoning": reason,
                "Choice": choice
            }
            
        except Exception as e:
            print(f"Response formatting failed: {str(e)}")
            raise ValueError("Could not format response according to MedQA standards")


    def _parse_section(self, text: str, header: str) -> str:
        """Strict section parser that enforces consistent formatting"""
        try:
            # For reasoning sections
            if "Select Optimal Treatment Plan" in header:
                pattern = r'(?:Select Optimal Treatment Plan|Reasoning):\s*(.*?)(?=\n\s*\d+\.|\nExpress Conclusion|\Z)'
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                return match.group(1).strip() if match else ""

            # For choice sections - enforce strict formatting
            elif "Choice" in header or "Express Conclusion" in header:
                # First try exact required format
                strict_pattern = r'Choice:\s*(\{[A-Z]\}:\s*\{[^}]+\})'
                match = re.search(strict_pattern, text)
                if match:
                    return match.group(1).strip()
                
                # If strict fails but has correct content, reformat it
                content_pattern = r'Choice:\s*([A-Z]):\s*([^\n]+)'
                match = re.search(content_pattern, text)
                if match:
                    option, med = match.groups()
                    return f"{{{option}}}: {{{med.strip(' .')}}}"
                
                return ""

            return ""
        except Exception:
            return ""

    def _get_relevant_history_safe(self, round_num: int, pool: 'HistoricalSharedPool') -> List[Dict]:
        """Retrieves comprehensive lead physician analysis from recent rounds"""
        try:
            if not hasattr(pool, 'get_round_statements'):
                return []

            max_rounds_to_show = 1 if round_num == 2 else 2  # Show 1 opinion in round 2, 2 for later rounds
            history = []
            
            # Get complete lead physician analysis from previous rounds
            for r in range(max(1, round_num - max_rounds_to_show), round_num):
                statements = pool.get_round_statements(r)
                if statements and len(statements) > 1:  # [0]=specialists, [1]=lead
                    lead = statements[1]
                    if isinstance(lead, dict) and lead.get("Agent_Name") == "Lead Physician":
                        history.append({
                            "round": r,
                            "consistency": lead.get("consistency", []),
                            "conflict": lead.get("conflict", []),
                            "independence": lead.get("independence", {}),
                            "integration": lead.get("integration", []),
                            "final_decision": lead.get("final_decision", "")
                        })
            
            return history[-max_rounds_to_show:]  # Return only the most recent analysis

        except Exception as e:
            print(f"History retrieval error: {str(e)}")
            return []
        
    def _create_error_response(self, round_num: int) -> Dict:
        """Creates an error response that maintains the expected structure"""
        return {
            "specialist": self.role,
            "Reasoning": "Error: Could not generate reason",
            "Choice": "Error: Could not get choice",

        }

