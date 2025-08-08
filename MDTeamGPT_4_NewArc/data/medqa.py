import random
import re
from datasets import load_dataset
from typing import Dict, Tuple, Optional

class OfficialMedQA:
    def __init__(self, dataset_name: str = "bigbio/med_qa"):
        """
        Load official MedQA dataset from HuggingFace
        Args:
            dataset_name: Name of dataset (default: "bigbio/med_qa")
        """
        self.current_index = 0  # Initialize the counter

        try:
            self.dataset = load_dataset(dataset_name)
            # Most BigBio datasets use 'train' split, but we'll verify
            self.questions = self.dataset['train'] if 'train' in self.dataset else list(self.dataset.values())[0]
            print(f"Successfully loaded {len(self.questions)} MedQA cases")
        except Exception as e:
            print(f"Error loading MedQA: {str(e)}")
            self.questions = []

   
    def get_case_by_number(self, question_num: int) -> Tuple[str, str, str, Dict]:
        """
        Get a specific case by question number (1-based index)
        Returns: (background, question with options, official_answer, metadata)
        """
        # Convert to 0-based index
        index = question_num - 1
        
        # Check bounds
        if index < 0 or index >= len(self.questions):
            print(f"Question number {question_num} out of range (1-{len(self.questions)})")
            return self._get_fallback_case()
        
        # Temporarily set current index
        original_index = self.current_index
        self.current_index = index
        
        # Get the case using existing method
        case = self.get_next_case()
        
        # Restore original index
        self.current_index = original_index
        
        return case

    def get_next_case(self) -> Tuple[str, str, str, Dict]:
   
        if not self.questions or self.current_index >= len(self.questions):
            return self._get_fallback_case()
        
        case = self.questions[self.current_index]
        self.current_index += 1  # Move to next question
        
        full_question = case.get('question', '')
        background, medical_problem = self._split_question(full_question)
        options_list = case.get('options', [])

        cleaned_options = self._format_options(options_list)
        cleaned_answer = self._clean_text(case.get('answer', ''))

        # Find the correct option key for the official answer
        correct_key = None
        for option in cleaned_options:
            if option['value'].lower() == cleaned_answer.lower():
                correct_key = option['key']
                break

        if correct_key:
            formatted_official_answer = f"{{{correct_key.upper()}}}: {{{cleaned_answer}}}"
        else:
            formatted_official_answer = f"{{?}}: {{{cleaned_answer}}}"
            
       
        return(
            full_question,
            cleaned_options,
            formatted_official_answer,
            {
                'question_id': case.get('id', str(self.current_index)),
                'options': cleaned_options,
                'context': case.get('context', ''),
                'index': self.current_index - 1,  # Return 0-based index
                'raw_medical_problem': medical_problem
            }
        )
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        # Remove all leading/trailing whitespace (including \n, \t)
        text = text.strip()
        # Remove triple quotes of either type (''' or """)
        text = re.sub(r'[\'"]{3}', '', text)
        # Remove any remaining leading/trailing single/double quotes
        text = text.strip('\'"')
        # Replace newlines and tabs with spaces
        text = re.sub(r'[\n\t]+', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _format_options(self, options) -> list:
        formatted = []
        
        if isinstance(options, dict):
            options = [{'key': k, 'value': v} for k, v in options.items()]
        elif not isinstance(options, list):
            return []
        
        for option in options:
            if isinstance(option, dict):
                # Clean both key and value
                cleaned = {
                    'key': self._clean_text(option.get('key', '')).upper(),
                    'value': self._clean_text(option.get('value', ''))
                }
                # Additional protection for JSON responses
                cleaned['value'] = cleaned['value'].replace('"', "'")  # Replace double with single quotes
                formatted.append(cleaned)
            else:
                # Handle non-dict options
                cleaned_value = self._clean_text(option)
                cleaned_value = cleaned_value.replace('"', "'")  # Replace double with single quotes
                formatted.append({
                    'key': '',
                    'value': cleaned_value
                })
        
        return formatted

    # def _clean_text(self, text: str) -> str:
    #     if not isinstance(text, str):
    #         text = str(text)
    #     # Remove all leading/trailing whitespace (including \n, \t)
    #     text = text.strip()
    #     # Remove triple quotes of either type (''' or """)
    #     text = re.sub(r'^[\'"]{3}|[\'"]{3}$', '', text)
    #     # Remove any remaining leading/trailing quotes
    #     text = text.strip('\'"')
    #     # Final whitespace clean in case quotes had internal spaces
    #     return text.strip()

    # def _format_options(self, options) -> list:
    #     formatted = []
        
    #     if isinstance(options, dict):
    #         options = [{'key': k, 'value': v} for k, v in options.items()]
    #     elif not isinstance(options, list):
    #         return []
        
    #     for option in options:
    #         if isinstance(option, dict):
    #             # Clean both key and value
    #             cleaned = {
    #                 'key': str(option.get('key', '')).strip().upper(),
    #                 'value': str(option.get('value', '')).strip()
    #             }
    #             # Remove trailing quotes and whitespace
    #             cleaned['value'] = cleaned['value'].rstrip('"\' \n\t')
    #             formatted.append(cleaned)
    #         else:
    #             # Handle non-dict options (convert to dict if needed)
    #             cleaned = {
    #                 'key': '',
    #                 'value': str(option).strip().rstrip('"\' \n\t')
    #             }
    #             formatted.append(cleaned)
        
    #     return formatted

   

    def get_case_by_id(self, question_id: str) -> Tuple[str, str, str, Dict]:
        """
        Get a specific case by question ID
        Returns: (background, question with options, official_answer, metadata)
        """
        for idx, question in enumerate(self.questions):
            if str(question.get('id', '')) == str(question_id):
                return self.get_case_by_number(idx + 1)
        
        print(f"Question ID {question_id} not found")
        return self._get_fallback_case()


    def _split_question(self, question: str) -> Tuple[str, str]:
        """
        Splits the full question into:
        - Patient background (clinical context)
        - Medical problem (actual question)
        """
        if not question:
            return "No background provided", "No question provided"
            
        # Common pattern in MedQA questions
        split_phrases = [
            "Which of the following",
            "What is the most likely",
            "The most likely diagnosis is"
        ]
        
        for phrase in split_phrases:
            if phrase in question:
                parts = question.split(phrase, 1)
                return parts[0].strip(), f"{phrase}{parts[1].strip()}"
        
        # Fallback: First sentence as background, rest as question
        sentences = question.split('.')
        return sentences[0].strip(), '.'.join(sentences[1:]).strip()

    def _get_fallback_case(self) -> Tuple[str, str, Dict]:
        """Fallback case when dataset isn't available"""
        return (
            "A 32-year-old man presents to the emergency department with a severe headache. "
            "He says that the pain has been getting progressively worse over the last 24 hours "
            "and is located primarily in his left forehead and eye. The headaches have woken him "
            "up from sleep and are not relieved by over-the-counter medications.",
            "Which of the following findings would most likely also be seen in this patient?",
            {
                'question_id': 'fallback_001',
                'options': [
                    {'key': 'A', 'value': 'Anosmia'},
                    {'key': 'B', 'value': 'Mandibular pain'},
                    {'key': 'C', 'value': 'Ophthalmoplegia'},
                    {'key': 'D', 'value': 'Vertigo'},
                    {'key': 'E', 'value': 'Vision loss'}
                ],
                'answer': 'C',
                'context': 'Imaging shows thrombosis of a sinus above the sella turcica.'
            }
        )
    def get_total_questions(self) -> int:
        """Return the total number of questions in the dataset"""
        return len(self.questions)

    