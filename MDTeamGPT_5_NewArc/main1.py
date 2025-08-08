import json
import time
import sys
import os
import random
from typing import List, Dict
from agents.primary_care import PrimaryCareDoctor
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

class MarkdownLogger:
    def __init__(self, log_file, original_stdout):
        self.log_file = log_file
        self.original_stdout = original_stdout
        self.start_time = datetime.now()
        self._write_header()
        
    def _write_header(self):
        """Write the initial markdown header"""
        self.log_file.write(f"# Medical Consultation Log\n\n")
        self.log_file.write(f"**Start Time:** {self.start_time.isoformat()}\n\n")
        self.log_file.write("## Processing Timeline\n\n")
        self.log_file.write("| Timestamp | Stage | Message |\n")
        self.log_file.write("|-----------|-------|---------|\n")
        self.log_file.flush()
        
    def write(self, message):
        # Write to original stdout (console)
        self.original_stdout.write(message)
        
        # Only log non-empty messages
        message = message.strip()
        if message:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-2]
            
            # Detect stage from message patterns 
            stage = "General"
            if "=====Step" in message:
                stage = message.split(":")[0].replace("=", "").strip()
            elif "=====" in message and "====" in message:
                stage = "Section Header"
            
            # Write to markdown file
            self.log_file.write(f"| {timestamp} | {stage} | `{message}` |\n")
            self.log_file.flush()
            
    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()
        
    def close(self):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.log_file.write("\n## Summary\n")
        self.log_file.write(f"- **Start Time:** {self.start_time.isoformat()}\n")
        self.log_file.write(f"- **End Time:** {end_time.isoformat()}\n")
        self.log_file.write(f"- **Total Duration:** {duration.total_seconds():.2f} seconds\n")
        
        self.flush()
        self.log_file.close()

def setup_logging(batch_num=None, total_batches=None):
    """Set up logging with batch information in filename using Markdown format"""
    os.makedirs("log", exist_ok=True)
    #os.makedirs("historical_pool", exist_ok=True)
    
    if batch_num is not None and total_batches is not None:
        log_filename = f"log/medqa_batch_{batch_num}-of-{total_batches}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    else:
        log_filename = f"log/medqa_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    log_file = open(log_filename, 'w+', encoding='utf-8')
    original_stdout = sys.stdout
    
    markdown_logger = MarkdownLogger(log_file, original_stdout)
    sys.stdout = markdown_logger
    
    return markdown_logger

def main():
    main_log_file = setup_logging()

    try:
        medqa = OfficialMedQA()
        total_questions = medqa.get_total_questions()
        target_cases = 1
        batch_size = 1  # Process 50 cases per batch
        
        #case_numbers = random.sample(range(1, total_questions + 1), min(target_cases, total_questions))
        #case_numbers.sort()  # Process in order
        case_numbers = [200]   
        #10: 200,214,303,415,524,620,663,706,717,738,
        #10: 815,865,993,956,975,1202,1264,1268,1291,1331,
        #10: 1424,1605,1619,1735,2151,2603,2717,2757,2834,2858
        #10: 3355,3391,3396,3423,3446,3813,3856,3956,4028,4288
        #10: 4325,4374,4586,5106,5376,5580,5863,5882,5927,5946

        #10: 7287,8591

        # very difficult question : 717
        
        total_batches = (len(case_numbers) + batch_size - 1) // batch_size
        all_results = []

        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(case_numbers))
            batch_cases = case_numbers[batch_start:batch_end]
            batch_log_file = setup_logging(batch_num + 1, total_batches)
            try:
                print(f"\n{'='*100}")
                print(f"Processing Batch {batch_num + 1}/{total_batches} (Cases {batch_cases[0]}-{batch_cases[-1]})")
                print(f"{'='*100}")

                # Process batch
                batch_results = process_batch_dataset(medqa,"MedQA",case_numbers=batch_cases)
                all_results.extend(batch_results)

                # print("\n====Each Batch Evaluation Metrics:=====")
                # calculate_evaluation_metrics(batch_results)
                     
            finally:
                sys.stdout = batch_log_file.original_stdout
                batch_log_file.close()
                print(f"\n**Batch log saved to:** `{batch_log_file.log_file.name}`")
            
    finally:
        sys.stdout = main_log_file.original_stdout
        main_log_file.close()
        print(f"\n**Main log saved to:** `{main_log_file.log_file.name}`")

def process_batch_dataset(dataset, dataset_name: str, case_numbers: list):
    results = []

    for i, q_num in enumerate(case_numbers, 1):
        try:
            print(f"Processing case {i}/{len(case_numbers)} (ID: {q_num})")
            question, options, answer, meta = dataset.get_case_by_number(q_num)
            
            start_time = time.time()
            result = run_consultation(
                q_num,
                question, 
                options, 
                answer, 
                meta                 
            )
            elapsed = time.time() - start_time
            
            # Store results
            result.update({
                "processing_time": elapsed,
                "timestamp": datetime.now().isoformat()
            })
            results.append(result)
            print(f"result: {result}")

        except Exception as e:
            print(f"Error processing case {q_num}: {str(e)}")
            results.append({"error": str(e)})
            
    return results

def run_consultation(
                q_num: int,
                question: str, 
                options: str,
                official_answer: str,
                metadata: Dict) -> Dict:
  
    print("="*100)
    print(f"============Starting Consultation for Case {q_num}============")
    
    # Extract agents from dict
    historical_pool = HistoricalSharedPool(case_id=q_num) 
    primary_care = PrimaryCareDoctor(historical_pool=historical_pool)
    specialist = SpecialistDoctor(historical_pool=historical_pool)
     # lead_physician = LeadPhysician()
    # chain_reviewer = ChainOfThoughtReviewer()
    # ethics_reviewer = SafetyEthicsReviewer()

    initial_data = {
    "question_id": q_num,
    "question": question,
    "options": options,
    "official_answer": official_answer,
    "timestamp": datetime.now().isoformat()
    }
    historical_pool.add_statements(initial_data)

#------------------------Step 1: Primary care assigns specialists-----------------------
    print("="*80)
    print(f"=====Step 1: Primary Care Agent =====")
    
    primary_care_response = primary_care.perform_task(question,options)
       
    #historical_pool.add_statements(primary_care_response)

#------------------------Step 2: Specialists provide opinions------------------------
    print("="*80)
    print(f"=====Step 2: Specialists Agents Discussion=====")

    specialist_respnse = specialist.perform_task(primary_care_response, question, options)
    print("\n====== Final consensused answer from all specialists ======\n")
    print(specialist_respnse)
    historical_pool.add_statements(specialist_respnse)

    # # New: Add LoRA validation
    # print("\n===== LoRA Model Validation =====")
    # validation = validator.validate_decision(question, specialist_respnse)
    # print(f"Model prediction: {validation['predicted_option']} (confidence: {validation['confidence']:.2f})")

    final_conclusion = specialist_respnse.get("final_conclusion", {})
    consensus_answer = final_conclusion.get("selected_option", {}).get("key")
    
    # # Flag disagreements
    # if validation['predicted_option'] != consensus_answer:
    #     print(f"⚠️ Warning: Model disagrees with consensus ({consensus_answer} vs {validation['predicted_option']})")


    return {
    "Question_ID": metadata.get("question_id"),
    "Final_Answer": final_conclusion.get("selected_option"),
    "Status":final_conclusion.get("status")
    # "Model_Validation": validation
    }


if __name__ == "__main__":
    main()

