import json
import time
import sys
import os
from typing import List, Dict
from collections import defaultdict
from agents.primary_care import PrimaryCareDoctor
from agents.specialist import SpecialistDoctor
from agents.lead_physician import LeadPhysician
from agents.chain_reviewer import ChainOfThoughtReviewer
from agents.safety_ethics import SafetyEthicsReviewer
from utils.shared_pool import HistoricalSharedPool
from data.medqa import OfficialMedQA
from data.pubmedqa import PubMedQA
from sklearn.metrics import f1_score,classification_report
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

def setup_logging(batch_start=None, batch_end=None):
    """Set up logging to both console and file with batch-specific naming"""
    # Create directories if they don't exist
    os.makedirs("log", exist_ok=True)
    os.makedirs("result", exist_ok=True)
    
    #if batch_start is not None and batch_end is not None:
    log_filename = f"log/medqa_batch_{batch_start}-{batch_end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    #else:
        #log_filename = f"log/medqa_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    # Create a Tee object that writes to both stdout and the log file
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    
    return log_file


def main():
    # Initialize main log file
    main_log_file = setup_logging()
    
    try:
        medqa = OfficialMedQA()
        total_questions = medqa.get_total_questions()
        batch_size = 20
    
        
        # Check for existing progress
        progress_file = "result/medqa_progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
            start_case = progress["last_processed"] + 1
            all_results = progress["results"]
            print(f"\nResuming from case {start_case} (previous progress found)")
            # Calculate metrics for already processed results
            print("\nMetrics for previously processed cases:")
            calculate_evaluation_metrics(all_results)
        else:
            start_case = 1
            all_results = []
            print("\nStarting new processing from case 1")

        # Process in batches
        for batch_start in range(start_case-1, total_questions, batch_size):
            batch_end = min(batch_start + batch_size, total_questions)
            
            # Set up batch-specific logging
            batch_log_file = setup_logging(batch_start+1, batch_end)
            try:
                print(f"\nProcessing batch: questions {batch_start+1} to {batch_end}")
                
                # Process current batch
                batch_results = process_batch_dataset(
                    medqa, 
                    "MedQA", 
                    start_q=batch_start+1, 
                    end_q=batch_end
                )
                all_results.extend(batch_results)
                
                # Save progress after each batch
                progress = {
                    "last_processed": batch_end,
                    "results": all_results
                }
                with open(progress_file, "w") as f:
                    json.dump(progress, f, indent=2)

                # Calculate metrics for current batch
                print(f"\nMetrics for batch {batch_start+1}-{batch_end}:")
                calculate_evaluation_metrics(batch_results)
                
                print(f"\nCompleted batch: questions {batch_start+1} to {batch_end}")
                
            finally:
                # Clean up logging for this batch
                sys.stdout = sys.stdout.files[0]
                batch_log_file.close()
                print(f"Batch log saved to: {batch_log_file.name}")

        # Save final results
        results_filename = f"result/medqa_full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, "w") as f:
            json.dump(all_results, f, indent=2)
            
        # Calculate final metrics
        print("\nFinal evaluation metrics:")
        calculate_evaluation_metrics(all_results)

        # Clean up progress file after successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
            
        print("\nProcessing complete!")
        print(f"Full results saved to: {results_filename}")
    
    except Exception as e:
        print(f"\nProcessing interrupted: {str(e)}")
        if 'all_results' in locals() and all_results:
            print("\nMetrics for processed cases before interruption:")
            calculate_evaluation_metrics(all_results)
        print(f"Progress saved. You can resume later.")
    finally:
        sys.stdout = sys.stdout.files[0]
        main_log_file.close()
        print(f"Main log saved to: {main_log_file.name}")


def process_batch_dataset(dataset, dataset_name: str, start_q: int, end_q: int):
    """Process a specific batch of questions in a dataset"""
    results = []
    total_in_batch = end_q - start_q + 1
    
    for q_num in range(start_q, end_q + 1):
        try:
            print(f"\n{'='*80}\nProcessing {dataset_name} question {q_num-start_q+1}/{total_in_batch} in batch (overall {q_num}/{dataset.get_total_questions()})")
            
            # Get the case data
            patient_background, medical_problem_with_options, official_answer, metadata = dataset.get_case_by_number(q_num)
            
            print(f"\n=== Case #{metadata.get('question_id', '')} ===")
            print(f"\nBackground: {patient_background[:200]}...")
            print(f"\nQuestion: {medical_problem_with_options[:200]}...")
            print(f"\nOfficial Answer: {official_answer}")
            
            # Run consultation
            start_time = time.time()
            result = run_consultation(patient_background, medical_problem_with_options, official_answer, metadata)
            elapsed = time.time() - start_time
            
            #

            # Store results
            result["processing_time"] = elapsed
            results.append(result)
            
            print(f"\nResult: Correct={result['is_correct']} | Time={elapsed:.2f}s")
            print(f"Final Decision: {result['final_decision']}")
                
        except Exception as e:
            print(f"Error processing question {q_num}: {str(e)}")
            continue
            
    return results

def process_all_dataset(dataset, dataset_name: str, max_questions: int = None):
    """Process all questions in a dataset"""
    results = []
    total = min(dataset.get_total_questions(), max_questions) if max_questions else dataset.get_total_questions()
    
    for q_num in range(1, total + 1):
        try:
            print(f"\n{'='*80}\nProcessing {dataset_name} question {q_num}/{total}")
            #bg, problem, answer, meta = dataset.get_next_case()
            patient_background, medical_problem_with_options,official_answer,metadata = dataset.get_next_case()
        
            print(f"\n=== Case #{metadata.get('question_id', '')} ===")
            print(f"\nBackground: {patient_background[:200]}...")
            print(f"\nQuestion: {medical_problem_with_options[:200]}...")
            print(f"\nOfficial Answer: {official_answer}")
            
            start_time = time.time()
            result = run_consultation(patient_background, medical_problem_with_options, official_answer, metadata)
            elapsed = time.time() - start_time
            
            result["processing_time"] = elapsed
            results.append(result)
            
            print(f"\nResult: Correct={result['is_correct']} | Time={elapsed:.2f}s")
            print(f"Final Decision: {result['final_decision']}")
            
            # Save periodic checkpoints
            if q_num % 5 == 0:
                with open(f"{dataset_name}_results_checkpoint.json", "w") as f:
                    json.dump(results, f)
                    
        except Exception as e:
            print(f"Error processing question {q_num}: {str(e)}")
            continue
            
    return results


def run_consultation(patient_background: str, 
                medical_problem_with_options: str,
                official_answer: str,
                metadata: Dict,
                max_rounds: int = 4) -> Dict:
    """Run full consultation pipeline for a single case"""
    # Step 1: Initialize all agents and components
    print("="*50)
    print("=====Step 1: Initializing Agents =====")
    primary_care = PrimaryCareDoctor()
    lead_physician = LeadPhysician()
    chain_reviewer = ChainOfThoughtReviewer()
    ethics_reviewer = SafetyEthicsReviewer()
    historical_pool = HistoricalSharedPool()

    print("All agents initialized successfully\n")
    
    # Step 2: Primary care assigns specialists
    print("="*50)
    print(f"=====Step 2: Primary Care Doctor Assigning Specialists =====")
    
    # perform task
    assignment = primary_care.perform_task(patient_background, medical_problem_with_options)
    #print(f"\nPrimary Care Assignment: {assignment}")

    selected_specialists = [
        SpecialistDoctor(spec_name)
        for spec_name in assignment.get("specialists", []) ]
    
    # Fallback if no specialists were assigned
    if not selected_specialists:
        print("Warning: No valid specialists assigned! Using General Practitioner")
        selected_specialists = [SpecialistDoctor("General Practitioner")]

    print(f"\n[System] Selected Specialists: {[s.specialty for s in selected_specialists]}")
    
    # Step 3: Specialists provide opinions
    print("="*50)
    print(f"=====Step 3: Specialists provide opinions=====")
    max_rounds = 4
    consensus_reached = False
    final_opinion = None
    correctKB, chainKB = load_knowledge_bases()
    
    for round_num in range(1, max_rounds + 1):
        time.sleep(1)
        print(f"\n=== Consultation Round {round_num}/{max_rounds} ===")
        
        # Show historical pool state
        print(f"[Historical Pool State] Current Rounds Stored: {historical_pool.current_round}")
        print(f"[Historical Pool State] All Statements:")
        for r, stmts in historical_pool.get_all_statements().items():
            print(f"  Round {r}:")
            #print(f"\n{stmts}:")

        # perform task
        specialist_opinions = []
        for specialist in selected_specialists:
        
            try:
                opinion = specialist.perform_task(
                    patient_background, 
                    medical_problem_with_options,
                    metadata.get('options',[]),
                    round_num, 
                    historical_pool,
                    correctKB,
                    chainKB
                )
                
                print(f"\n===Specialist Output {specialist.role} Opinion===")
                print(json.dumps(opinion, indent=4))    
                # Store each specialist's opinion in the format you want
                specialist_opinions.append({
                    "Agent_Name": opinion.get("Agent_Name", specialist.role),
                    "Reasoning": opinion.get("Reasoning", ""),
                    "Choice": opinion.get("Choice", "")
                })
            except Exception as e:
                print(f"  - CRITICAL ERROR: {str(e)}")
                specialist_opinions.append({
                    "Agent_Name": specialist.role,
                    "Reasoning": f"Error: {str(e)}",
                    "Choice": "Unable to generate"
                })          
        round_data = {
            f"round {round_num}": specialist_opinions
        }
        print(f"\n===Add specialist opinions to historical pool===")
        historical_pool.add_statements(round_data)

                
        # Step 4: Lead physician analyzes opinions
        print("="*50)
        print("=====Step 4:Lead Physician Analyzing opinions=====")
    
        # perform task
        leadphysician_opinions = []
        analysis = lead_physician.perform_task(specialist_opinions)
        print(f"\n====Lead physician Output Opinion=====")
        #print(json.dumps(analysis, indent=4))  
        leadphysician_opinions = {
            f"round {round_num}": {
                "Agent_Name": "Lead Physician",
                "consistency": analysis.get("consistency", []),
                "conflict": analysis.get("conflict", []),
                "independence": analysis.get("independence", []),
                "integration": analysis.get("integration", [])
            }
        }

        # Store in historical pool
        print(f"\n===Add lead physician opinions to historical pool===")
        historical_pool.add_statements(leadphysician_opinions)
        
        # Check consensus
        print("\n===Consensus Check===")
        all_choices = [opinion.get("Choice", "").strip() for opinion in specialist_opinions]
        print(f"Current round choices: {all_choices}")

        # Method 1: Exact match consensus (all choices identical)
        if all(choice == all_choices[0] for choice in all_choices):
            consensus_reached = True
            final_opinion = all_choices[0]
            #print(f"\n{final_opinion}")
            print(f"Consensus reached: {all_choices[0]}")
            break
        
        # Method 2: Majority vote if final round
        if round_num == max_rounds:
            print("  >> Final round reached - determining majority choice")
            
            # Count votes for each choice
            choice_counts = {}
            for choice in all_choices:
                if choice:  # Only count non-empty choices
                    choice_counts[choice] = choice_counts.get(choice, 0) + 1
            
            if choice_counts:
                # Get choice with highest count
                majority_choice = max(choice_counts.items(), key=lambda x: x[1])[0]
                majority_count = choice_counts[majority_choice]
                total_votes = sum(choice_counts.values())
                
                print(f"  >> Voting results:")
                for choice, count in choice_counts.items():
                    print(f"     - {choice}: {count}/{total_votes} votes")
                
                final_opinion = majority_choice
                print(f"  >> Selected majority choice: {majority_choice} ({majority_count}/{total_votes} votes)")
            else:
                final_opinion = "No valid choices provided by specialists"
                print("  >> No valid choices to vote on")

    # Step 5: Safety and ethics review
    print("="*50)
    print("=====Step 5:Ethics Reviewer=====")

    # perform task
    ethics_review = ethics_reviewer.perform_task(
        final_opinion, 
        patient_background,
        historical_pool
    )

    print("\n=== Final Ethics Review Result ===")
    if ethics_review.get("approved", True):  # Default to True if key missing
        print("✅ APPROVED: The treatment plan meets safety and ethical standards")
        if ethics_review.get("concerns"):
            print("\n⚠️ Note: Some concerns were identified (treatment still approved):")
            for concern in ethics_review.get("concerns", []):
                print(f"- {concern}")
    else:
        print("❌ NOT APPROVED: Significant safety or ethical issues found")
        print("\nPrimary concerns:")
        for concern in ethics_review.get("concerns", ["No specific concerns listed"]):
            print(f"- {concern}")
        
        print("\nRecommendations:")
        for rec in ethics_review.get("recommendations", ["Consult with ethics committee"]):
            print(f"- {rec}")

    # Show assessment summary if available
    if "assessment" in ethics_review:
        print(f"\nSummary: {ethics_review['assessment']}")

    # Step 6: Chain of thought reviewer
    print("="*50)
    print("=====Step 6:Chain of thought Review=====")

    review_result = chain_reviewer.perform_task(
        final_opinion=final_opinion,
        official_answer=official_answer,
        patient_background=patient_background,
        medical_problem_with_options=medical_problem_with_options,
        historical_pool=historical_pool.get_all_statements(),
        metadata={
            "specialists": [spec.role for spec in selected_specialists],
            "rounds": historical_pool.current_round,
            **metadata  # Include all original metadata
        }
    )
    print(f"Chain-of-thought processing result: {review_result}") 
    
    return {
    "question_id": metadata.get("question_id"),
    "final_decision": final_opinion,
    "correct_answer": official_answer,
    "is_correct": review_result.get("correctness", False),
    "ethics_review": ethics_review,
    "rounds": historical_pool.current_round,
    "specialists": [spec.role for spec in selected_specialists]
    }


def calculate_evaluation_metrics(results):
    """Calculate and print evaluation metrics for the results"""
    if not results:
        print("No results to evaluate")
        return
    
    # Extract predictions and ground truth
    y_true = []
    y_pred = []
    
    for result in results:
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
    print(f"\nBasic Metrics:")
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
            
            print("\nClassification Report:")
            print(classification_report(
                y_true_encoded, y_pred_encoded, 
                target_names=le.classes_, 
                zero_division=0
            ))
        except Exception as e:
            print(f"\nCould not calculate F1 scores: {str(e)}")
    else:
        print("\nNot enough class diversity for F1 calculation")
    
    # Additional simple metrics
    correct_bool = sum(1 for result in results if result['is_correct'])
    total = len(results)
    print(f"\nCorrect (bool): {correct_bool}/{total} ({correct_bool/total:.2%})")

def load_knowledge_bases():
    """Load correct and chain KBs from JSON files"""
    vector_db_path = Path("vector_db_storage")
    
    correctKB = []
    chainKB = []
    
    try:
        with open(vector_db_path / "correct_answers.json", "r") as f:
            correctKB = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: Could not load correct_answers.json")
    
    try:
        with open(vector_db_path / "chain_of_thought.json", "r") as f:
            chainKB = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: Could not load chain_of_thought.json")
    
    return correctKB, chainKB

if __name__ == "__main__":
    main()

