#!/usr/bin/env python3
"""
BioMedLM LoRA Fine-tuning Script - Production Ready
Optimized for Mac M2/M3, single GPU, and multi-GPU setups
"""

import os
import json
import torch
import warnings
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset, Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)

# Suppress warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

@dataclass
class ModelArguments:
    model_name_or_path: str = "stanford-crfm/BioMedLM"
    use_4bit_quantization: bool = True  # Enable by default for memory efficiency
    use_fp16: bool = True
    use_mps: bool = True  # Enable MPS for Mac M2/M3

@dataclass
class DataArguments:
    dataset_name: str = "Malikeh1375/medical-question-answering-datasets"
    dataset_config: str = "all-processed"
    max_length: int = 512
    train_split_ratio: float = 0.9
    max_eval_samples: int = 1000  # NEW: Limit evaluation dataset size

@dataclass
class LoRAArguments:
    r: int = 16  # Rank - can increase to 32 for better performance
    lora_alpha: int = 32  # Alpha parameter
    target_modules: List[str] = field(default_factory=lambda: ["c_proj", "c_attn"])  # More modules
    lora_dropout: float = 0.1
    bias: str = "none"

class BioMedLMTrainer:
    def __init__(self, model_args: ModelArguments, data_args: DataArguments, 
                 training_args: TrainingArguments, lora_args: LoRAArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.lora_args = lora_args
        
        # Smart device selection
        self.device = self._get_optimal_device()
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def _get_optimal_device(self):
        """Select the best available device"""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self.model_args.use_mps:
            return "mps"
        else:
            return "cpu"
    
    def setup_model_and_tokenizer(self):
        """Load and setup the model with LoRA configuration"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path, 
            trust_remote_code=True,
            padding_side="right"  # Important for training
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("Loading model...")
        
        # Configure quantization for memory efficiency
        quantization_config = None
        if self.model_args.use_4bit_quantization and self.device.startswith('cuda'):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("Using 4-bit quantization")
        
        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16 if self.model_args.use_fp16 else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        elif self.device != "cpu":
            model_kwargs["device_map"] = "auto"
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with device_map: {e}")
            # Fallback without device_map
            model_kwargs.pop("device_map", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                **model_kwargs
            )
        
        # Prepare model for training
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Configure LoRA with improved target modules detection
        target_modules = self._get_target_modules()
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_args.r,
            lora_alpha=self.lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.bias,
        )
        
        print(f"LoRA target modules: {target_modules}")
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Move to device if not using quantization
        if not quantization_config and self.device != "cpu":
            self.model = self.model.to(self.device)
    
    def _get_target_modules(self):
        """Automatically detect target modules for LoRA"""
        # Get all module names
        module_names = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_names.append(name.split('.')[-1])
        
        # Common linear layer names in different architectures
        common_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_proj", "c_attn", "c_fc"]
        
        # Find matching modules
        target_modules = list(set(module_names) & set(common_targets))
        
        # Fallback to specified modules if auto-detection fails
        if not target_modules:
            target_modules = self.lora_args.target_modules
            print("Warning: Auto-detection failed, using default target modules")
        
        return target_modules
        
    def load_and_prepare_dataset(self):
        """Load and preprocess the dataset with better error handling"""
        print("Loading dataset...")
        
        try:
            dataset = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config)
            
            # Use the train split or the first available split
            if 'train' in dataset:
                data = dataset['train']
            else:
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]
                print(f"Using split: {split_name}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to a smaller subset for testing...")
            
            # Create a small dummy dataset for testing
            dummy_data = [
                {"question": "What are the symptoms of diabetes?", "answer": "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision."},
                {"question": "How is hypertension treated?", "answer": "Treatment may include lifestyle changes and medications like ACE inhibitors or diuretics."},
                {"question": "What causes pneumonia?", "answer": "Pneumonia can be caused by bacteria, viruses, or fungi affecting the lungs."},
            ] * 100  # Repeat for more training examples
            
            data = Dataset.from_list(dummy_data)
        
        print(f"Dataset loaded with {len(data)} examples")
        print("Dataset columns:", data.column_names)
        print("Sample data:", data[0])
        
        # Prepare the dataset for training with improved formatting
        def format_prompt(example):
            """Format the example into question-answer format"""
            # Try different field combinations
            field_mappings = [
                ('question', 'answer'),
                ('input', 'output'),
                ('prompt', 'response'),
                ('text', 'target'),
            ]
            
            question, answer = None, None
            
            # Try exact matches first
            for q_field, a_field in field_mappings:
                if q_field in example and a_field in example:
                    question = str(example[q_field]).strip()
                    answer = str(example[a_field]).strip()
                    break
            
            # Try partial matches
            if not question or not answer:
                fields = list(example.keys())
                question_fields = [f for f in fields if any(kw in f.lower() for kw in ['question', 'input', 'prompt'])]
                answer_fields = [f for f in fields if any(kw in f.lower() for kw in ['answer', 'output', 'response', 'target'])]
                
                if question_fields and answer_fields:
                    question = str(example[question_fields[0]]).strip()
                    answer = str(example[answer_fields[0]]).strip()
                elif len(fields) >= 2:
                    question = str(example[fields[0]]).strip()
                    answer = str(example[fields[1]]).strip()
                else:
                    # Single field - assume pre-formatted
                    return {"text": str(example[fields[0]]).strip()}
            
            # Format with special tokens for better structure
            formatted_text = f"### Question: {question}\n### Answer: {answer}{self.tokenizer.eos_token}"
            return {"text": formatted_text}
        
        # Apply formatting
        formatted_data = data.map(
            format_prompt, 
            remove_columns=data.column_names,
            desc="Formatting examples"
        )
        
        # Split into train and validation
        split_data = formatted_data.train_test_split(
            test_size=1 - self.data_args.train_split_ratio,
            seed=42
        )
        
        train_dataset = split_data['train']
        eval_dataset = split_data['test']
        
        # OPTIMIZATION 1: Limit evaluation dataset size for faster evaluation
        if len(eval_dataset) > self.data_args.max_eval_samples:
            print(f"Limiting eval dataset from {len(eval_dataset)} to {self.data_args.max_eval_samples} samples for faster evaluation")
            # Use select with shuffling to get a representative subset
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(self.data_args.max_eval_samples))
        
        print(f"Train examples: {len(train_dataset)}")
        print(f"Eval examples: {len(eval_dataset)} (limited for MPS performance)")
        
        # Tokenize the datasets
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.data_args.max_length,
                return_tensors=None,
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing train dataset",
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing eval dataset",
        )
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        return True
    
    def train(self):
        """Execute the training process"""
        print("Starting training...")
        
        # Data collator with dynamic padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Create output directory
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        
        # Save training configuration
        config = {
            "model_args": self.model_args.__dict__,
            "data_args": self.data_args.__dict__,
            "lora_args": self.lora_args.__dict__,
            "training_args": {k: str(v) for k, v in self.training_args.__dict__.items()},
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(os.path.join(self.training_args.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Train the model
        try:
            trainer.train()
            
            # Save the final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.training_args.output_dir)
            print(f"Model saved to: {self.training_args.output_dir}")
            
            # Save training metrics
            if trainer.state.log_history:
                with open(os.path.join(self.training_args.output_dir, "training_metrics.json"), "w") as f:
                    json.dump(trainer.state.log_history, f, indent=2, default=str)
                    
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True

def create_training_arguments(output_dir: str = "./biomedlm-lora-finetuned", 
                            device: str = "cpu") -> TrainingArguments:
    """Create properly configured training arguments with version compatibility and MPS optimization"""
    
    # Check transformers version for parameter compatibility
    import transformers
    transformers_version = transformers.__version__
    print(f"Transformers version: {transformers_version}")
    
    # Base configuration
    base_args = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 4,  # OPTIMIZATION 3: Larger eval batch size
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,  # This will be overridden by evaluation_strategy="epoch"
        "save_strategy": "steps",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
    }
    
    # OPTIMIZATION 2: Evaluate only once per epoch instead of every few steps
    # Handle evaluation_strategy vs eval_strategy compatibility
    try:
        # Try with newer parameter name first
        test_args = base_args.copy()
        test_args["evaluation_strategy"] = "epoch"  # Changed from "steps" to "epoch"
        TrainingArguments(**{k: v for k, v in test_args.items() if k in ["output_dir", "evaluation_strategy"]})
        base_args["evaluation_strategy"] = "epoch"
        print("Using 'evaluation_strategy=epoch' parameter for less frequent evaluation")
    except TypeError:
        # Fall back to older parameter name
        base_args["eval_strategy"] = "epoch"  # Changed from "steps" to "epoch"
        print("Using 'eval_strategy=epoch' parameter (older transformers version)")
    
    # Device-specific optimizations
    if device.startswith("cuda"):
        base_args.update({
            "fp16": True,
            "dataloader_pin_memory": True,
            "per_device_train_batch_size": 2,  # Can be larger on GPU
            "per_device_eval_batch_size": 8,   # Even larger for eval on GPU
        })
    elif device == "mps":
        base_args.update({
            "fp16": False,  # MPS can be unstable with fp16 in some transformers versions
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 4,  # Optimized for MPS
        })
        print("MPS optimizations applied: fp16=False, eval_batch_size=4, evaluation_strategy=epoch")
    else:  # CPU
        base_args.update({
            "fp16": False,  # CPU doesn't support fp16 well
            "dataloader_num_workers": 2,
            "per_device_eval_batch_size": 2,  # Conservative for CPU
        })
    
    try:
        return TrainingArguments(**base_args)
    except Exception as e:
        print(f"Error creating TrainingArguments: {e}")
        # Ultra-minimal fallback
        minimal_args = {
            "output_dir": output_dir,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 2,
            "num_train_epochs": 2,
            "logging_steps": 50,
            "save_steps": 1000,
            "fp16": False,
        }
        return TrainingArguments(**minimal_args)

def run_inference(model_path: str, question: str, max_length: int = 256, device: str = "auto"):
    """Run inference with the fine-tuned model"""
    print(f"Loading fine-tuned model from: {model_path}")
    
    # Auto-detect device if not specified
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model_kwargs = {
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "stanford-crfm/BioMedLM",
        **model_kwargs
    )
    
    # Load fine-tuned model
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    if device != "cpu":
        model = model.to(device)
    
    # Format input
    prompt = f"### Question: {question}\n### Answer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    
    return answer

def main():
    """Main training function"""
    print("BioMedLM LoRA Fine-tuning Script - Production Ready (MPS Optimized)")
    print("=" * 70)
    
    # Configure arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    lora_args = LoRAArguments()
    
    # Create a temporary trainer to get device info
    temp_trainer = BioMedLMTrainer(model_args, data_args, None, lora_args)
    device = temp_trainer.device
    
    # Create properly configured training arguments
    training_args = create_training_arguments(device=device)
    
    print(f"Configuration:")
    print(f"- Device: {device}")
    print(f"- 4-bit quantization: {model_args.use_4bit_quantization}")
    print(f"- FP16: {training_args.fp16}")
    print(f"- Train batch size: {training_args.per_device_train_batch_size}")
    print(f"- Eval batch size: {training_args.per_device_eval_batch_size}")
    print(f"- Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"- Learning rate: {training_args.learning_rate}")
    print(f"- Max eval samples: {data_args.max_eval_samples}")
    
    # Check evaluation strategy
    eval_strategy = getattr(training_args, 'evaluation_strategy', None) or getattr(training_args, 'eval_strategy', None)
    print(f"- Evaluation strategy: {eval_strategy}")
    
    # Initialize trainer
    trainer = BioMedLMTrainer(model_args, data_args, training_args, lora_args)
    
    # Setup model and tokenizer
    try:
        trainer.setup_model_and_tokenizer()
    except Exception as e:
        print(f"Failed to setup model: {e}")
        return
    
    # Load and prepare dataset
    if not trainer.load_and_prepare_dataset():
        print("Failed to load dataset. Exiting.")
        return
    
    # Start training
    success = trainer.train()
    
    if success:
        print("\nTraining completed successfully!")
        print(f"Model saved to: {training_args.output_dir}")
        
        # Demonstrate inference
        print("\n" + "="*60)
        print("INFERENCE DEMONSTRATION")
        print("="*60)
        
        test_questions = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "What causes pneumonia?"
        ]
        
        for question in test_questions:
            try:
                answer = run_inference(training_args.output_dir, question, device=device)
                print(f"\nQuestion: {question}")
                print(f"Answer: {answer}")
                print("-" * 40)
            except Exception as e:
                print(f"Inference error for '{question}': {e}")
                break
    else:
        print("Training failed.")

if __name__ == "__main__":
    # Set environment variables for optimal performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    
    # For CUDA memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()

# ============================================================================
# USAGE INSTRUCTIONS:
# ============================================================================
# 
# 1. Install required packages:
#    pip install torch transformers datasets peft bitsandbytes accelerate
#
# 2. For Mac M2/M3 (MPS support):
#    pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# 3. Run the script:
#    python biomedlm_lora_finetuning_optimized.py
#
# 4. For multi-GPU training:
#    torchrun --nproc_per_node=2 biomedlm_lora_finetuning_optimized.py
#
# 5. Monitor training:
#    - Check terminal output for progress
#    - Training metrics saved to training_metrics.json
#    - Model checkpoints saved every 500 steps
#
# ============================================================================
#
# MPS OPTIMIZATION CHANGES:
# ============================================================================
# 1. Added max_eval_samples=1000 to DataArguments to limit eval dataset size
# 2. Changed evaluation_strategy from "steps" to "epoch" for less frequent eval
# 3. Increased per_device_eval_batch_size to 4 for faster evaluation
# 4. Added logic to automatically limit eval dataset if it exceeds max_eval_samples
# 5. Enhanced device-specific optimizations for MPS backend
# ============================================================================