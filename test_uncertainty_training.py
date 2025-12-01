#!/usr/bin/env python3
"""
Test Script for ARD-LoRA Training with Uncertainty Evaluation
============================================================

This script demonstrates the enhanced training pipeline that:
1. Splits validation data into ARD prior estimation and uncertainty evaluation
2. Runs uncertainty evaluation (ACC, ECE, NLL) after each epoch
3. Logs results to tensorboard and saves JSON summaries

Usage:
    python test_uncertainty_training.py
    
    # With custom parameters
    python test_uncertainty_training.py --epochs 3 --samples 2000
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_uncertainty_training(epochs=2, uncertainty_samples=500, dataset="alpaca"):
    """Test the uncertainty evaluation training pipeline."""
    
    print("🚀 Testing ARD-LoRA Training with Uncertainty Evaluation")
    print("=" * 60)
    
    # Override configuration for testing
    from config import CONFIG
    
    test_config = {
        "defaults": {
            "model_name": "LLaMA2-7B",
            "dataset_name": "BayesianPEFT",
            "dataset_name_specific": dataset,
            "runId": 999,  # Test run
            "rank": 16,  # Smaller rank for faster testing
            "max_len": 512,  # Shorter sequences
            "train_epochs": epochs,
            "batch_size": 2,  # Small batch size
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "kl_loss_beta": 0.01,
            "beta": 0.01,
            "ard_prior_ratio": 0.6,  # 60% for ARD, 40% for uncertainty
            "uncertainty_eval_samples": uncertainty_samples,
            "uncertainty_n_bins": 10,  # Fewer bins for small test
            "fp16": False,  # Avoid potential issues
            "load_in_4bit": False,
            "report_to": ["tensorboard"]
        }
    }
    
    # Temporarily override config
    original_config = CONFIG.copy() if CONFIG else {}
    CONFIG.clear()
    CONFIG.update(test_config)
    
    try:
        # Import and run training
        from run_training_cached import main
        
        print(f"📊 Test Configuration:")
        print(f"   Dataset: {dataset}")
        print(f"   Epochs: {epochs}")
        print(f"   Uncertainty samples: {uncertainty_samples}")
        print(f"   ARD/Uncertainty split: 60/40")
        print(f"   Rank: 16 (reduced for testing)")
        print(f"   Max length: 512 (reduced for testing)")
        
        print(f"\n🏃 Starting training...")
        main()
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Check outputs/ARD_LoRA_LLaMA2-7B_{dataset}_999/ for results")
        print(f"📊 Uncertainty results saved in uncertainty_results.json")
        print(f"📈 Tensorboard logs available for uncertainty metrics")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original config
        CONFIG.clear()
        CONFIG.update(original_config)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test ARD-LoRA training with uncertainty evaluation')
    parser.add_argument('--epochs', type=int, default=2, 
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of samples for uncertainty evaluation (default: 500)')
    parser.add_argument('--dataset', type=str, default='alpaca',
                       choices=['alpaca', 'alpaca_gpt4', 'dolly', 'oasst1'],
                       help='Dataset to use for testing (default: alpaca)')
    
    args = parser.parse_args()
    
    test_uncertainty_training(
        epochs=args.epochs,
        uncertainty_samples=args.samples,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()