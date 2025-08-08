import random
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

    def get_next_case(self) -> Tuple[str, str, str, Dict]:
        
        if not self.questions or self.current_index >= len(self.questions):
            return self._get_fallback_case()
        
        case = self.questions[self.current_index]
        self.current_index += 1  # Move to next question
        
        full_question = case.get('question', '')
        background, medical_problem = self._split_question(full_question)
        official_answer = case.get('answer', '')
        options_list = case.get('options', [])

        # Find the correct option key for the official answer
        correct_key = None
        for option in options_list:
            if option['value'].lower() == official_answer.lower():
                correct_key = option['key']
                break

        # Format the official answer with the correct key
        if correct_key:
            formatted_official_answer = f"{{{correct_key.upper()}}}: {{{official_answer}}}"
        else:
            formatted_official_answer = f"{{?}}: {{{official_answer}}}"
            
        options = self._format_options(options_list)
        medical_problem_with_options = f"{medical_problem}\n\nOptions:\n{options}"
        
        return (
            full_question,
            options,
            formatted_official_answer,
            {
                'question_id': case.get('id', str(self.current_index)),
                'options': options,
                'context': case.get('context', ''),
                'index': self.current_index - 1,  # Return 0-based index
                'raw_medical_problem': medical_problem
            }
        )
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

    def _format_options(self, options) -> list:
        """Ensure options are in consistent format"""
        if isinstance(options, dict):
            return [{'key': k, 'value': v} for k, v in options.items()]
        elif isinstance(options, list):
            return options
        return []

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

    