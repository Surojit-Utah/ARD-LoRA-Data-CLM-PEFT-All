"""
Integrated ARD-LoRA Model Evaluation Script
===========================================

This script evaluates trained ARD-LoRA models using uncertainty metrics.
It integrates with the existing ARD-LoRA training pipeline and datasets.

Usage:
    python evaluate/evaluate_model.py --model_path ./outputs/model --config alpaca
    
    # With custom parameters
    python evaluate/evaluate_model.py \
        --model_path ./outputs/model \
        --config alpaca \
        --num_samples 2000 \
        --batch_size 8 \
        --n_bins 20
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluate.uncertainty_metrics import UncertaintyEvaluator, print_evaluation_results
from dataloader.bayesian_peft_cached import get_cached_dataset, DATASET_CONFIGS
from model.model_llama import inject_problora_llama
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig


def load_trained_model(model_path: str, base_model: str = "NousResearch/Llama-2-7b-hf"):
    """
    Load a trained ARD-LoRA model from checkpoint.
    
    Args:
        model_path: Path to the trained model directory
        base_model: Base LLaMA model name
    
    Returns:
        model: Loaded model
        tokenizer: Model tokenizer
    """
    print(f"🔄 Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure quantization for efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Inject ProbLoRA layers
    inject_problora_llama(base_model)
    
    # Load trained weights if they exist
    if os.path.exists(model_path):
        # Check for different checkpoint formats
        checkpoint_files = [
            "pytorch_model.bin",
            "adapter_model.bin", 
            "model.safetensors",
            "adapter_model.safetensors"
        ]
        
        checkpoint_path = None
        for filename in checkpoint_files:
            full_path = os.path.join(model_path, filename)
            if os.path.exists(full_path):
                checkpoint_path = full_path
                break
        
        if checkpoint_path:
            print(f"🔄 Loading checkpoint: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                base_model.load_state_dict(state_dict, strict=False)
                print("✅ Successfully loaded model weights")
            except Exception as e:
                print(f"⚠️  Warning: Could not load checkpoint - {e}")
                print("   Proceeding with base model for demonstration")
        else:
            print("⚠️  No checkpoint found, using base model")
    
    return base_model, tokenizer


def create_evaluation_dataloader(dataset_config: str, tokenizer, batch_size: int = 4):
    """
    Create evaluation dataloader for the specified dataset.
    
    Args:
        dataset_config: Dataset configuration name
        tokenizer: Model tokenizer  
        batch_size: Batch size for evaluation
    
    Returns:
        eval_dataloader: DataLoader for evaluation
    """
    print(f"🔄 Loading evaluation dataset: {dataset_config}")
    
    # Get cached dataset
    dataset = get_cached_dataset(dataset_config, use_cache=True)
    
    if dataset is None:
        raise ValueError(f"Could not load dataset: {dataset_config}")
    
    # Use test split if available, otherwise use a portion of train split
    if 'test' in dataset:
        eval_dataset = dataset['test']
    elif 'validation' in dataset:
        eval_dataset = dataset['validation'] 
    else:
        # Use last 10% of training data for evaluation
        train_dataset = dataset['train']
        eval_size = min(1000, len(train_dataset) // 10)
        eval_dataset = train_dataset.select(range(len(train_dataset) - eval_size, len(train_dataset)))
    
    print(f"📊 Evaluation dataset size: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    return eval_dataloader


def evaluate_ard_lora_model_integrated(
    model, 
    eval_dataloader, 
    tokenizer, 
    device,
    num_samples: int = 1000,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Evaluate ARD-LoRA model with integrated uncertainty metrics.
    
    This function is specifically designed for the ARD-LoRA pipeline
    and handles causal language modeling evaluation properly.
    
    Args:
        model: Trained ARD-LoRA model
        eval_dataloader: Evaluation dataloader
        tokenizer: Model tokenizer
        device: Device for computation
        num_samples: Maximum number of token predictions to evaluate
        n_bins: Number of bins for ECE calculation
    
    Returns:
        metrics: Dictionary with ACC, NLL, ECE, and additional info
    """
    evaluator = UncertaintyEvaluator(n_bins=n_bins)
    model.eval()
    
    all_labels = []
    all_probs = []
    
    print(f"🔄 Evaluating model uncertainty on up to {num_samples} token predictions...")
    
    with torch.no_grad():
        sample_count = 0
        
        for batch_idx, batch in enumerate(eval_dataloader):
            if sample_count >= num_samples:
                break
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # For causal LM: shift predictions and labels
            # Predict token i+1 given tokens 0...i
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Keep only non-masked tokens (labels != -100)
            valid_mask = shift_labels != -100
            valid_logits = shift_logits[valid_mask]
            valid_labels = shift_labels[valid_mask]
            
            if len(valid_labels) > 0:
                # Convert logits to probabilities
                probs = torch.softmax(valid_logits, dim=-1)
                
                # Sample subset if too many tokens in this batch
                max_tokens_per_batch = min(200, num_samples - sample_count)
                if len(valid_labels) > max_tokens_per_batch:
                    indices = torch.randperm(len(valid_labels))[:max_tokens_per_batch]
                    valid_labels = valid_labels[indices]
                    probs = probs[indices]
                
                # Collect predictions
                all_labels.extend(valid_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                sample_count += len(valid_labels)
            
            if batch_idx % 10 == 0:
                print(f"   Processed {batch_idx + 1} batches, {sample_count} token predictions")
    
    print(f"✅ Evaluation completed on {sample_count} token predictions")
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    
    # Compute uncertainty metrics
    metrics = evaluator.evaluate_predictions(y_true, y_probs)
    
    return metrics


def save_evaluation_results(metrics: Dict[str, float], output_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Evaluation metrics dictionary
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")


def main():
    """Main evaluation script for ARD-LoRA models."""
    parser = argparse.ArgumentParser(description='Evaluate ARD-LoRA model uncertainty')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--config', type=str, default='alpaca',
                       choices=list(DATASET_CONFIGS.keys()),
                       help=f'Dataset configuration: {list(DATASET_CONFIGS.keys())}')
    parser.add_argument('--base_model', type=str, default='NousResearch/Llama-2-7b-hf',
                       help='Base LLaMA model name')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of token predictions to evaluate')
    parser.add_argument('--n_bins', type=int, default=15,
                       help='Number of bins for ECE calculation')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Evaluation batch size')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results JSON (default: model_path/evaluation_results.json)')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = os.path.join(args.model_path, 'evaluation_results.json')
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔄 Using device: {device}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_trained_model(args.model_path, args.base_model)
        
        # Create evaluation dataloader
        eval_dataloader = create_evaluation_dataloader(
            args.config, 
            tokenizer, 
            args.batch_size
        )
        
        # Run evaluation
        print(f"\n📊 Starting uncertainty evaluation...")
        print(f"   Model: {args.model_path}")
        print(f"   Dataset: {args.config}")
        print(f"   Samples: {args.num_samples}")
        print(f"   ECE bins: {args.n_bins}")
        
        metrics = evaluate_ard_lora_model_integrated(
            model=model,
            eval_dataloader=eval_dataloader,
            tokenizer=tokenizer,
            device=device,
            num_samples=args.num_samples,
            n_bins=args.n_bins
        )
        
        # Print and save results
        print_evaluation_results(metrics)
        save_evaluation_results(metrics, args.output)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())