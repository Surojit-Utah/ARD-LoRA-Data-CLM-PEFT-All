#!/usr/bin/env python3
"""
Test script to verify that validation split creation works properly
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dataloader.bayesian_peft_cached import load_bayesian_peft_with_caching

def test_validation_split():
    """Test that validation split is created when missing"""
    print("Testing validation split creation...")
    
    # Test configuration
    config = {
        "dataset_name": "alpaca",
        "max_len": 512,
        "cache_root": "./test_cache",
        "batch_size": 2,
        "num_labels": 0
    }
    
    try:
        # Load dataset
        train_ds, val_ds, tokenizer = load_bayesian_peft_with_caching(
            dataset_name="alpaca",
            tokenizer_name="gpt2",  # Use a simple tokenizer for testing
            config=config,
            cache_root="./test_cache"
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Training samples: {len(train_ds) if train_ds else 0}")
        print(f"   Validation samples: {len(val_ds) if val_ds else 0}")
        
        if val_ds is None or len(val_ds) == 0:
            print("❌ No validation data created!")
            return False
        else:
            print("✅ Validation split created successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test cache
        import shutil
        if os.path.exists("./test_cache"):
            shutil.rmtree("./test_cache")
            print("🧹 Cleaned up test cache")

if __name__ == "__main__":
    success = test_validation_split()
    sys.exit(0 if success else 1)