"""
Uncertainty Evaluation Package for ARD-LoRA Models
==================================================

This package provides standard uncertainty evaluation metrics for assessing
the quality of uncertainty estimation in ARD-LoRA models:

- Accuracy (ACC): Classification accuracy
- Expected Calibration Error (ECE): Measures confidence calibration quality
- Negative Log-Likelihood (NLL): Measures predictive probability quality

Usage:
    from evaluate import UncertaintyEvaluator
    
    evaluator = UncertaintyEvaluator(n_bins=15)
    metrics = evaluator.evaluate_predictions(y_true, y_probs)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"NLL: {metrics['nll']:.4f}")
"""

from .uncertainty_metrics import UncertaintyEvaluator, evaluate_ard_lora_model, print_evaluation_results

__version__ = "1.0.0"
__author__ = "ARD-LoRA Research Team"

__all__ = [
    'UncertaintyEvaluator',
    'evaluate_ard_lora_model', 
    'print_evaluation_results'
]