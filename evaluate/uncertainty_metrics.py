"""
Uncertainty Evaluation Metrics for ARD-LoRA Models
==================================================

This module implements standard uncertainty evaluation metrics:
- Accuracy (ACC): Classification accuracy
- Expected Calibration Error (ECE): Confidence calibration quality  
- Negative Log-Likelihood (NLL): Predictive probability quality

Usage:
    python -m evaluate.uncertainty_metrics --model_path ./outputs/model --dataset_path ./data
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, List
import argparse
import os
from pathlib import Path


class UncertaintyEvaluator:
    """Evaluator for uncertainty estimation metrics on ARD-LoRA models."""
    
    def __init__(self, n_bins: int = 15):
        """
        Initialize uncertainty evaluator.
        
        Args:
            n_bins: Number of bins for ECE calculation (default: 15)
        """
        self.n_bins = n_bins
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Args:
            y_true: Ground truth labels (N,)
            y_pred: Predicted labels (N,)
        
        Returns:
            acc: Accuracy score [0, 1]
        """
        return accuracy_score(y_true, y_pred)
    
    def compute_nll(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Compute Negative Log-Likelihood (NLL).
        
        NLL = -1/N * Σ log P_θ(y_n)
        
        This metric prefers models that assign higher probabilities to correct labels.
        Lower NLL indicates better predictive probability quality.
        
        Args:
            y_true: Ground truth labels (N,)
            y_probs: Predicted probabilities (N, C)
        
        Returns:
            nll: Negative log-likelihood
        """
        # Get probabilities for true labels
        true_probs = y_probs[np.arange(len(y_true)), y_true]
        
        # Avoid log(0) by clipping to small epsilon
        true_probs = np.clip(true_probs, 1e-8, 1.0)
        
        # Calculate negative log-likelihood
        nll = -np.mean(np.log(true_probs))
        
        return nll
    
    def compute_ece(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ (|B_m|/N) * |acc(B_m) - conf(B_m)|
        
        Where:
        - B_m is the set of samples whose confidence falls in bin m
        - |B_m| is the number of samples in bin m
        - acc(B_m) is the accuracy of samples in bin m
        - conf(B_m) is the average confidence of samples in bin m
        
        This measures how well the model's confidence matches its accuracy.
        Lower ECE indicates better calibration.
        
        Args:
            y_true: Ground truth labels (N,)
            y_probs: Predicted probabilities (N, C)
        
        Returns:
            ece: Expected calibration error [0, 1]
        """
        # Get predicted labels and max probabilities (confidence)
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        # Create bin boundaries
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in current bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Compute accuracy and confidence in this bin
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add weighted difference to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluate_predictions(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """
        Compute all uncertainty metrics for given predictions.
        
        Args:
            y_true: Ground truth labels (N,)
            y_probs: Predicted probabilities (N, C)
        
        Returns:
            metrics: Dictionary containing all metrics
        """
        y_pred = np.argmax(y_probs, axis=1)
        
        metrics = {
            'accuracy': self.compute_accuracy(y_true, y_pred),
            'nll': self.compute_nll(y_true, y_probs),
            'ece': self.compute_ece(y_true, y_probs),
            'num_samples': len(y_true),
            'num_classes': y_probs.shape[1]
        }
        
        return metrics


def evaluate_ard_lora_model(model, eval_dataloader, tokenizer, device, 
                           num_samples: int = 1000, n_bins: int = 15) -> Dict[str, float]:
    """
    Evaluate ARD-LoRA model using uncertainty metrics for causal language modeling.
    
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
                probs = F.softmax(valid_logits, dim=-1)
                
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


def print_evaluation_results(metrics: Dict[str, float]):
    """
    Print evaluation results in a formatted way.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 60)
    print("📊 UNCERTAINTY EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"📈 **Accuracy (ACC)**:     {metrics['accuracy']:.4f}")
    print(f"   → Higher is better (perfect = 1.0)")
    print()
    
    print(f"📉 **Negative Log-Likelihood (NLL)**: {metrics['nll']:.4f}")
    print(f"   → Lower is better (perfect = 0.0)")
    print(f"   → Measures predictive probability quality")
    print()
    
    print(f"🎯 **Expected Calibration Error (ECE)**: {metrics['ece']:.4f}")
    print(f"   → Lower is better (perfect = 0.0)")
    print(f"   → Measures confidence calibration quality")
    print()
    
    print(f"📊 **Dataset Info**:")
    print(f"   → Evaluated samples: {metrics['num_samples']:,}")
    print(f"   → Vocabulary size: {metrics['num_classes']:,}")
    print("=" * 60)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate ARD-LoRA model uncertainty')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--dataset_config', type=str, default='alpaca',
                       help='Dataset configuration name')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of token predictions to evaluate')
    parser.add_argument('--n_bins', type=int, default=15,
                       help='Number of bins for ECE calculation')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🔄 Using device: {device}")
    
    # This would be implemented to load your specific model and dataset
    print(f"🔄 Loading model from: {args.model_path}")
    print(f"🔄 Dataset: {args.dataset_config}")
    print(f"🔄 Evaluation samples: {args.num_samples}")
    print(f"🔄 ECE bins: {args.n_bins}")
    
    # Placeholder for actual model and data loading
    # You would implement this based on your specific setup
    print("⚠️  Model and dataset loading not implemented in this demo script")
    print("   Integrate with your ARD-LoRA model loading code")
    
    # Example usage with dummy data
    print("\n🧪 Running example with synthetic data...")
    
    # Create synthetic evaluation data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 1000  # Vocabulary size
    
    # Generate synthetic predictions (poorly calibrated for demonstration)
    y_true = np.random.randint(0, n_classes, n_samples)
    logits = np.random.randn(n_samples, n_classes) * 2.0
    y_probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Evaluate with synthetic data
    evaluator = UncertaintyEvaluator(n_bins=args.n_bins)
    metrics = evaluator.evaluate_predictions(y_true, y_probs)
    
    print_evaluation_results(metrics)
    
    print(f"\n💡 To use with real models, integrate with your model loading code")
    print(f"   See evaluate_ard_lora_model() function for reference")


if __name__ == "__main__":
    main()