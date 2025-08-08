from datasets import load_dataset
from typing import Dict, Tuple, Optional
import random

class PubMedQA:
    def __init__(self, dataset_name: str = "pubmed_qa"):
        """
        Load official PubMedQA dataset
        Options: 
        - 'pubmed_qa' (main dataset)
        - 'pubmed_qa_labeled_fold0' to 'fold4' (5-fold cross validation sets)
        - 'pubmed_qa_artificial' (synthetic questions)
        """
        try:
            self.dataset = load_dataset("pubmed_qa", dataset_name)
            self.questions = self.dataset['train']  # Using training split
        except Exception as e:
            print(f"Error loading PubMedQA: {str(e)}")
            self.questions = []

    def get_random_case(self) -> Tuple[str, str, Optional[str], Dict]:
        """
        Returns:
        - (context, question, answer, metadata)
        """
        if not self.questions:
            return self._get_fallback_case()
        
        case = random.choice(self.questions)
        return (
            case['context'],
            case['question'],
            case.get('long_answer', None),
            {
                'answer': case.get('final_decision', 'unknown'),
                'labels': case.get('labels', []),
                'meshes': case.get('meshes', []),
                'pubid': case.get('pubid', ''),
                'type': 'pubmed_qa'
            }
        )

    def get_case_by_pubid(self, pubid: str) -> Optional[Tuple[str, str, str, Dict]]:
        """Get specific case by PubMed ID"""
        for case in self.questions:
            if case.get('pubid') == pubid:
                return (
                    case['context'],
                    case['question'],
                    case.get('long_answer', None),
                    {
                        'answer': case['final_decision'],
                        'labels': case.get('labels', []),
                        'meshes': case.get('meshes', []),
                        'pubid': pubid
                    }
                )
        return None

    def _get_fallback_case(self) -> Tuple[str, str, str, Dict]:
        """Fallback case if dataset fails to load"""
        return (
            "A 45-year-old male presents with chest pain radiating to the left arm...",
            "What is the most likely diagnosis?",
            "The symptoms are consistent with acute myocardial infarction...",
            {
                'answer': 'mi',
                'labels': ['cardiology', 'emergency'],
                'meshes': ['Myocardial Infarction'],
                'pubid': 'fallback123',
                'type': 'fallback'
            }
        )