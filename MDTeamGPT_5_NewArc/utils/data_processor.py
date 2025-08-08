import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DataProcessor:
    @staticmethod
    def split_data(input_path: str, output_dir: str, test_size=0.2):
        """Split MedQA into train/test sets (80/20)"""
        with open(input_path) as f:
            data = json.load(f)
        
        train, test = train_test_split(data, test_size=test_size, random_state=42)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{output_dir}/train.json", "w") as f:
            json.dump(train, f)
        with open(f"{output_dir}/test.json", "w") as f:
            json.dump(test, f)
        print(f"Split complete: {len(train)} train, {len(test)} test cases")

    @staticmethod
    def group_by_specialist(input_path: str, config_path: str):
        """Group training data by specialist using keywords"""
        with open(input_path) as f:
            train_data = json.load(f)
        with open(config_path) as f:
            specialist_cfg = json.load(f)
        
        grouped = defaultdict(list)
        for case in train_data:
            question = case["question"].lower()
            for spec, keywords in specialist_cfg.items():
                if any(kw in question for kw in keywords):
                    grouped[spec].append(case)
                    break
            else:
                grouped["general"].append(case)
        
        for spec, cases in grouped.items():
            with open(f"{Path(input_path).parent}/train_{spec}.json", "w") as f:
                json.dump(cases, f)
        print(f"Grouped into {len(grouped)} specialists")