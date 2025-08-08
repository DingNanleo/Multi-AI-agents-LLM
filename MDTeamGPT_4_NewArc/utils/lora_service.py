# lora_service.py
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from pathlib import Path
import json
import os

class LoRADecisionValidator:
    def __init__(self, model_path="models/lora_medical"):
        os.makedirs(model_path, exist_ok=True) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        

        # Load with better error handling
        try:
            self.base_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            if Path(model_path).exists():
                self._load_trained_model()
            else:
                print(f"Warning: Model path {model_path} not found. Using untrained model.")
                self._init_untrained_model()
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def load_specialist_adapter(self, specialist: str):
        """Dynamically load a specialist's LoRA"""
        adapter_path = f"adapters/{specialist}"
        if not Path(adapter_path).exists():
            raise ValueError(f"No adapter found for {specialist}")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)

    def _load_trained_model(self):
        """Load trained LoRA model"""
        config_path = Path(self.model_path) / "adapter_config.json"
        model_path = Path(self.model_path) / "adapter_model.bin"
        classifier_path = Path(self.model_path) / "classifier.pt"
        
        if not (config_path.exists() and model_path.exists() and classifier_path.exists()):
            print(f"Warning: Model files missing in {self.model_path}. Initializing new model.")
            self._init_untrained_model()
            return
        
        self.model = PeftModel.from_pretrained(self.base_model, self.model_path).to(self.device)
        self.classifier = torch.nn.Linear(768, 5).to(self.device)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.model.eval()
    
    def _init_untrained_model(self):
        """Initialize for training"""
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.model = get_peft_model(self.base_model, lora_config).to(self.device)
        self.classifier = torch.nn.Linear(768, 5).to(self.device)

    def validate_decision(self, question, specialist_responses):
        """Check if specialist consensus aligns with model prediction"""
        input_text = self._format_input(question, specialist_responses)
        
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled)
            probs = torch.softmax(logits, dim=-1)
            
            predicted = chr(ord('A') + torch.argmax(probs).item())
            confidence = torch.max(probs).item()
            
            return {
                "predicted_option": predicted,
                "confidence": confidence,
                "probabilities": probs.cpu().numpy().tolist()
            }
    
    def _format_input(self, question, responses):
        """Improved formatting for specialist responses"""
        parts = [f"Question: {question}"]
        
        # Handle both direct responses and nested 'round_x' format
        if isinstance(responses, dict) and "final_conclusion" in responses:
            # New consultation format
            for key, data in responses.items():
                if key.startswith("round_") and isinstance(data, dict):
                    parts.append(f"{data['specialist']}: {data['selected_option']['key']} - {data['reasoning']}")
        else:
            # Historical log format
            for role, data in responses.items():
                if isinstance(data, dict) and "selected_option" in data:
                    parts.append(f"{role}: {data['selected_option']['key']} - {data['reasoning']}")
        
        return "\n".join(parts)

# Singleton pattern for easy access
validator = LoRADecisionValidator()