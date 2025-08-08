import json
import time
import sys
import os
import random
from typing import List, Dict
from agents.primary_care import PrimaryCareDoctor
print(f"PrimaryCareDoctor class: {PrimaryCareDoctor.__module__}.{PrimaryCareDoctor.__name__}")
from agents.specialist import SpecialistDoctor
from agents.lead_physician import LeadPhysician
from agents.chain_reviewer import ChainOfThoughtReviewer
from agents.safety_ethics import SafetyEthicsReviewer
from utils.shared_pool import HistoricalSharedPool
from data.medqa import OfficialMedQA
from data.pubmedqa import PubMedQA
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from pathlib import Path

class Tee:
    """Class to duplicate output to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate write
    
    def flush(self):
        for f in self.files:
            f.flush()

def setup_logging(batch_num=None, total_batches=None):
    """Set up logging with batch information in filename"""
    os.makedirs("log", exist_ok=True)
    os.makedirs("historical_pool", exist_ok=True)
    
    if batch_num is not None and total_batches is not None:
        log_filename = f"log/medqa_batch_{batch_num}-of-{total_batches}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    else:
        log_filename = f"log/medqa_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    
    return log_file

def main():
    main_log_file = setup_logging()

    try:
        print("=====Initializing Agents for Batch=====")
        primary_care = PrimaryCareDoctor()
        lead_physician = LeadPhysician()
        chain_reviewer = ChainOfThoughtReviewer()
        safety_ethics = SafetyEthicsReviewer()
        print("=====Initializing Agents Finished=====")

        medqa = OfficialMedQA()
        total_questions = medqa.get_total_questions()
        target_cases = 1
        batch_size = 1 # Process 50 cases per batch
        
        #case_numbers = random.sample(range(1, total_questions + 1), min(target_cases, total_questions))
        #case_numbers.sort()  # Process in order
        case_numbers = [7287]   #7287, 8591
        
        total_batches = (len(case_numbers) + batch_size - 1) // batch_size
        all_results = []

        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(case_numbers))
            batch_cases = case_numbers[batch_start:batch_end]
            batch_log_file = setup_logging(batch_num + 1, total_batches)
            # the log can log the print since this code 
            try:
                print(f"\n{'='*100}")
                print(f"Processing Batch {batch_num + 1}/{total_batches} (Cases {batch_cases[0]}-{batch_cases[-1]})")
                print(f"{'='*100}")

                # Process batch
                batch_results = process_batch_dataset(
                    medqa, 
                    "MedQA", 
                    case_numbers=batch_cases,
                    agents={
                        'primary_care': primary_care,
                        'lead_physician': lead_physician,
                        'chain_reviewer': chain_reviewer,
                        'safety_ethics': safety_ethics
                    }
                )
                all_results.extend(batch_results)

                print("\n====Each Batch Evaluation Metrics:=====")
                calculate_evaluation_metrics(batch_results)
                
                
            finally:
                sys.stdout = sys.stdout.files[0]
                batch_log_file.close()
                print(f"Batch log saved to: {batch_log_file.name}")
            
        print("\n====Final Evaluation Metrics:====")
        calculate_evaluation_metrics(all_results)
    
    except Exception as e:
        print(f"\nProcessing interrupted: {str(e)}")
        if 'all_results' in locals() and all_results:
            print("\nMetrics for processed cases before interruption:")
            calculate_evaluation_metrics(all_results)
    finally:
        sys.stdout = sys.stdout.files[0]
        main_log_file.close()
        print(f"Main log saved to: {main_log_file.name}")

def process_batch_dataset(dataset, dataset_name: str, case_numbers: list, agents:Dict):
    results = []

    for i, q_num in enumerate(case_numbers, 1):
        try:
            print(f"Processing case {i}/{len(case_numbers)} (ID: {q_num})")
            historical_pool = HistoricalSharedPool(case_id=q_num) 
            question, options, answer, meta = dataset.get_case_by_number(q_num)
            
            start_time = time.time()
            result = run_consultation(
                q_num,
                question, 
                options, 
                answer, 
                meta,
                agents={
                    **agents,
                    'historical_pool': historical_pool
                }                  
            )
            elapsed = time.time() - start_time
            
            # Store results
            result.update({
                "processing_time": elapsed,
                "timestamp": datetime.now().isoformat()
            })
            results.append(result)
            
            print(f"Result: {'CORRECT' if result['is_correct'] else 'INCORRECT'} | Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"Error processing case {q_num}: {str(e)}")
            results.append({"error": str(e)})
            
    return results

def run_consultation(
                q_num: int,
                question: str, 
                options: str,
                official_answer: str,
                metadata: Dict,
                agents:Dict,
                max_rounds: int = 3) -> Dict:
  
    print("="*100)
    print(f"============Starting Consultation for Case {q_num}============")
    
    # Extract agents from dict
    primary_care = agents['primary_care']
    lead_physician = agents['lead_physician']
    safety_ethics = agents['safety_ethics'] 
    chain_reviewer = agents['chain_reviewer']
    historical_pool = agents['historical_pool']
    historical_pool.case_id = q_num 

    initial_data = {
    "question_id": q_num,
    "question": question,
    "options": options,
    "timestamp": datetime.now().isoformat()
    }
    historical_pool.add_statements(initial_data)
    #print(f"\n===Add questions bacground to historical pool===")

#------------------------Step 1: Primary care assigns specialists-----------------------
    print("="*80)
    print(f"=====Step 1: Primary Care Agent =====")
    
    while True:
        assignment = primary_care.perform_task(
            question, 
            options
            )
        #print(f"\nPrimary Care Assignment: {assignment}")

        selected_specialists = [
            SpecialistDoctor(spec_name)
            for spec_name in assignment.get("specialists", []) ]


        if selected_specialists and len(selected_specialists) >= 2:
            break
        print("Warning: No valid specialists assigned! Consult primary care again")

    print(f"\n[System] Selected Specialists: {[s.specialty for s in selected_specialists]}")
    
    primarycare_data = {
        "primary_care_assignment": {
            "assigned_specialists": [spec.role for spec in selected_specialists]
        }
    }
    historical_pool.add_statements(primarycare_data)

#------------------------Step 2: Specialists provide opinions------------------------
    print("="*80)
    print(f"=====Step 2: Specialists Ageent=====")
    final_opinion = None
    majority_choice = None

    for round_num in range(1, max_rounds + 1):
        time.sleep(1)
        print(f"\n=== Consultation Round {round_num}/{max_rounds} ===")
        lastround_history = None

        if round_num > 1:
            lastround_history=historical_pool.get_round_statements(1)
            #print(f"history for last rond : {lastround_history}")

        specialist_opinions = []
        for specialist in selected_specialists:

            try:
                opinion = specialist.perform_task(
                    q_num,
                    question, 
                    options,
                    round_num,
                    lastround_history,
                    majority_choice
                )  
                specialist_opinions.append({
                    "Agent_Name": opinion.get("Agent_Name", specialist.role),
                    "Reason": opinion.get("Reason", ""),
                    "Choice": opinion.get("Choice", "")
                })
            except Exception as e:
                print(f"  - CRITICAL ERROR: {str(e)}")
                specialist_opinions.append({
                    "Agent_Name": specialist.role,
                    "Reason": f"Error: {str(e)}",
                    "Choice": "Unable to generate"
                })          
        round_data = {
            f"round {round_num}": {  
                "specialist_opinions": specialist_opinions
            }
        }
        historical_pool.add_statements(round_data)
                
#------------------------Step 3: Lead physician analyzes opinions------------------------
        # print("="*80)
        # print("=====Step 3:Lead Physician Agent=====")
    
        # leadphysician_opinions = []
        # analysis = lead_physician.perform_task(specialist_opinions)
        # #print(f"\n====Lead physician Output Opinion=====")
        # #print(json.dumps(analysis, indent=4))  

        # leadphysician_opinions = {
        #     f"round {round_num}": {
        #         "Lead_Physician_Opinion": {
        #             "Agent_Name": "Lead Physician",
        #             **analysis
        #         }
        #     }
        # }
        # historical_pool.add_statements(leadphysician_opinions)

#------------------------Step 4: Check Consensus------------------------
        def extract_key(choice_str):
            """Extract just the key portion from a 'key: value' string"""
            if ': ' in choice_str:
                return choice_str.split(': ', 1)[0].strip(' {}')
            return None 

        def similar(a, b):
            """Calculate similarity ratio between two strings"""
            return SequenceMatcher(None, a, b).ratio()
        
        print("\n============Consensus Check============")
        all_choices = [opinion.get("Choice", "").strip() for opinion in specialist_opinions]
        print(f"Current round choices: {all_choices}")

        invalid_choices = {
            "Unable to parse choice",
            "No valid choice format found",
            "Error:",
            "Unable to generate"
        }

        # Method 1: Exact match consensus (all choices identical AND no errors)
        if all(choice == all_choices[0] for choice in all_choices):
            first_choice = all_choices[0]
            if any(invalid in first_choice for invalid in invalid_choices):
                print(f"Invalid consensus (parsing errors): {first_choice} - continuing to next method")
            else:
                final_opinion = first_choice
                print(f"Valid consensus reached: {first_choice}")
                break
        
        # Method 2: Similarity-based consensus (ONLY when all keys match)
        else:
            # Extract keys from all choices
            keys = [extract_key(choice) for choice in all_choices]
            unique_keys = set(keys)
            
            # Only proceed if ALL keys are identical
            if len(unique_keys) == 1 and None not in unique_keys:
                current_key = unique_keys.pop()
                print(f"All keys match ({current_key}), comparing values...")
                
                # Extract just the values for comparison
                values = [choice.split(': ', 1)[1].strip(' {}') for choice in all_choices]
                
                # Calculate pairwise similarity
                similarity_scores = []
                for i in range(len(values)):
                    for j in range(i+1, len(values)):
                        similarity_scores.append(similar(values[i], values[j]))
                
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                print(f"Average value similarity: {avg_similarity:.2f}")
                
                if avg_similarity > 0.95:
                    # Return the most common exact phrasing
                    most_common = max(set(all_choices), key=all_choices.count)
                    final_opinion = most_common
                    
                    round_result = {
                        f"round {round_num}": {
                            "round_result": most_common,
                            "consensus_type": "similarity_based",
                            "average_similarity": avg_similarity
                        }
                    }
                    historical_pool.add_statements(round_result)

                    print(f"Consensus reached with similar values: {most_common}")
                    break
            else:
                choice_counts = {}
                for choice in all_choices:
                    if choice:  # Only count non-empty choices
                        choice_counts[choice] = choice_counts.get(choice, 0) + 1
                
                if choice_counts:
                    majority_choice = max(choice_counts.items(), key=lambda x: x[1])[0]
                    majority_count = choice_counts[majority_choice]
                    total_votes = sum(choice_counts.values())
                    if round_num == max_rounds:
                        print(f"  >> Voting results:")
                        for choice, count in choice_counts.items():
                            print(f"     - {choice}: {count}/{total_votes} votes")
                        final_opinion = majority_choice

                        round_result = {
                            f"round {round_num}": {
                                "round_result": majority_choice,
                                "consensus_type": "majority_vote",
                                "vote_count": majority_count,
                                "total_votes": total_votes,
                                "vote_percentage": majority_count/total_votes
                            }
                        }
                        historical_pool.add_statements(round_result)

                        print(f"  >> Selected majority choice: {majority_choice} ({majority_count}/{total_votes} votes)")
                        break
                else:
                    final_opinion = "No valid choices provided by specialists"
                    round_result = {
                        f"round {round_num}": {
                            "round_result": final_opinion,
                            "consensus_type": "no_valid_choices"
                        }
                    }
                    historical_pool.add_statements(round_result)
                    print("  >> No valid choices to vote on")
    
#------------------------Step 4: Safety and ethics review------------------------
    print("="*80)
    print("=====Step 4:Ethics Reviewer=====")

    total_history=historical_pool.get_all_statements()

    safety_ethics_opinion = safety_ethics.perform_task(
        final_opinion, 
        question,
        options,
        total_history
    )
    # ethics_answer = safety_ethics_opinion.get("recommended_answer", "")
    # if isinstance(ethics_answer, dict):
    #     # Convert {"C": "Gastric ulceration"} to "{C}: {Gastric ulceration}"
    #     key = next(iter(ethics_answer))  # Get first key
    #     ethics_answer = f"{{{key}}}: {{{ethics_answer[key]}}}"

    ethics_answer = safety_ethics_opinion.get("recommended_answer", "")
    if isinstance(ethics_answer, dict):
        ethics_answer = str(ethics_answer)

    reformatted_ethics_opinion = {
        "Safety_Ethics_Opinion": {
            "Agent_Name": "Safety Ethics",
            "Specialists_Answer": final_opinion,
            "Ethics_Answer": ethics_answer,
            "Ethics_Conclusion": safety_ethics_opinion.get("conclusion", "")
        }
    }
    historical_pool.add_statements(reformatted_ethics_opinion)

#------------------------Step 5: Chain of thought reviewer------------------------
    print("="*80)
    print("=====Step 5:Chain of thought Review=====")

    is_correct = compare_with_official_answer(final_opinion, official_answer)

    review_result = chain_reviewer.perform_task(
        final_opinion,
        official_answer,
        is_correct,
        question,
        options,
        total_history,
        metadata={
            "specialists": [spec.role for spec in selected_specialists],
            "round": historical_pool.current_round,
            **metadata  # Include all original metadata
        }
    )

    if is_correct:
        reformatted_chainreview_opinion = {
            "Chain_Reviewer_Opinion": {
                "Agent_Name": "Chain_Reviewer",
                "Question": review_result.get("Question", ""),
                "Answer": review_result.get("Answer", ""),
                "Summary": review_result.get("Summary", "")
            }
            }
    else:
        reformatted_chainreview_opinion = {
            "Chain_Reviewer_Opinion": {
                "Agent_Name": "Chain_Reviewer",
                "Correct Answer": review_result.get("Correct Answer", ""),
                "Initial Hypothesis ": review_result.get("Initial Hypothesis ", ""),
                "Analysis Process": review_result.get("Analysis Process ", ""),
                "Final Conclusion": review_result.get("Final Conclusion", ""),
                "Error Reflection": review_result.get("Error Reflection", "")
            }
            }
        
    historical_pool.add_statements(reformatted_chainreview_opinion)

    return {
    "question_id": metadata.get("question_id"),
    "final_decision": final_opinion,
    "correct_answer": official_answer,
    "is_correct": is_correct
    }

def compare_with_official_answer(final_opinion: str, official_answer: str) -> bool:
    # Simple comparison - you might want to implement more sophisticated matching
    return final_opinion.lower().strip() in official_answer.lower().strip() or \
            official_answer.lower().strip() in final_opinion.lower().strip()
    
def calculate_evaluation_metrics(results):
    """Calculate and print evaluation metrics for the results"""
    if not results:
        print("No results to evaluate")
        return
    #print(results)
    # Extract predictions and ground truth
    y_true = []
    y_pred = []
    
    for result in results:
        print("\n =====each result:=====")
        print(result)
        # Clean and standardize the answers
        true_answer = str(result['correct_answer']).strip().upper()
        pred_answer = str(result['final_decision']).strip().upper()
        
        # Extract just the letter (A, B, C, etc.) if present
        true_letter = true_answer[0] if true_answer and true_answer[0].isalpha() else true_answer
        pred_letter = pred_answer[0] if pred_answer and pred_answer[0].isalpha() else pred_answer
        
        y_true.append(true_letter)
        y_pred.append(pred_letter)
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0
    print(f"\n Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(y_true)} correct)")
    
    # Only calculate F1 if we have multiple classes
    unique_classes = set(y_true)
    if len(unique_classes) > 1:
        try:
            le = LabelEncoder()
            # Fit on all possible classes we might encounter
            all_possible_classes = set(y_true + y_pred)
            le.fit(list(all_possible_classes))
            
            y_true_encoded = le.transform(y_true)
            y_pred_encoded = le.transform(y_pred)
            
            macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
            micro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='micro', zero_division=0)
            
            print(f"Macro-F1: {macro_f1:.4f}")
            print(f"Micro-F1: {micro_f1:.4f}")
        except Exception as e:
            print(f"\nCould not calculate F1 scores: {str(e)}")
    else:
        print("\nNot enough class diversity for F1 calculation")
    
    # Additional simple metric
    correct_bool = sum(1 for result in results if result['is_correct'])
    total = len(results)
    print(f"Correct (bool): {correct_bool}/{total} ({correct_bool/total:.2%})")


if __name__ == "__main__":
    main()

