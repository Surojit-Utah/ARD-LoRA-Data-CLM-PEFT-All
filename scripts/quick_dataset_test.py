"""
Quick Dataset Test Script
=========================

A simplified version of test_dataset_sampling.py for rapid testing.
Tests all datasets with minimal samples and displays key information.

Usage:
    python scripts/quick_dataset_test.py
    python scripts/quick_dataset_test.py --dataset arc_easy
    python scripts/quick_dataset_test.py --model meta-llama/Llama-2-7b-hf
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
from transformers import AutoTokenizer
from dataset.utils import dsets


def test_dataset(dataset_name, dataset_class, init_params, tokenizer):
    """Quick test of a dataset."""
    print(f"\n{'='*70}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*70}")
    
    try:
        # Initialize dataset
        dataset_obj = dataset_class(tokenizer=tokenizer, add_space=True, **init_params)
        
        # Show target mappings
        print("\nTarget Token Mappings:")
        for label_idx, token_id in dataset_obj.label2target.items():
            token_str = tokenizer.convert_ids_to_tokens(token_id.item())
            print(f"  {label_idx} -> {repr(token_str)} (id: {token_id.item()})")
        
        # Test train split with 1 example
        if 'train' in dataset_obj.dset:
            train_split = dataset_obj.dset['train']
            if len(train_split) > 0:
                batch = [train_split[0]]
                prompts, classes, targets = dataset_obj.clm_collate_fn(batch)
                
                print(f"\nSample from TRAIN split ({len(train_split)} total):")
                print(f"  Prompt: {tokenizer.decode(prompts['input_ids'][0], skip_special_tokens=True)[:200]}...")
                print(f"  Label: {classes[0].item() if torch.is_tensor(classes[0]) else classes[0]}")
                print(f"  Target: {tokenizer.decode(targets[0])}")
                print(f"  ✓ Train split OK")
            else:
                print(f"  ⚠ Train split is empty")
        
        # Test validation split
        val_split_name = 'validation' if 'validation' in dataset_obj.dset else 'test'
        if val_split_name in dataset_obj.dset:
            val_split = dataset_obj.dset[val_split_name]
            if len(val_split) > 0:
                batch = [val_split[0]]
                prompts, classes, targets = dataset_obj.clm_collate_fn(batch)
                
                print(f"\nSample from {val_split_name.upper()} split ({len(val_split)} total):")
                print(f"  Prompt: {tokenizer.decode(prompts['input_ids'][0], skip_special_tokens=True)[:200]}...")
                print(f"  Label: {classes[0].item() if torch.is_tensor(classes[0]) else classes[0]}")
                print(f"  Target: {tokenizer.decode(targets[0])}")
                print(f"  ✓ {val_split_name.capitalize()} split OK")
            else:
                print(f"  ⚠ {val_split_name.capitalize()} split is empty")
        
        print(f"\n✓ {dataset_name} PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ {dataset_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick tests on all datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick dataset testing")
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', help='Model/tokenizer')
    parser.add_argument('--dataset', default='all', help='Specific dataset to test (default: all)')
    args = parser.parse_args()
    
    # Define datasets to test
    datasets_to_test = {
        'winogrande_s': (dsets.WinograndeDataset, {'name': 'winogrande_s'}),
        'winogrande_m': (dsets.WinograndeDataset, {'name': 'winogrande_m'}),
        'arc_easy': (dsets.ARCDataset, {'name': 'ARC-Easy'}),
        'arc_challenge': (dsets.ARCDataset, {'name': 'ARC-Challenge'}),
        'obqa': (dsets.OBQADataset, {}),
        'rte': (dsets.RTEDataset, {}),
        'mrpc': (dsets.MRPCDataset, {}),
        'cola': (dsets.CoLADataset, {}),
        'boolq': (dsets.BoolQDataset, {}),
    }
    
    print("\n" + "="*70)
    print("QUICK DATASET TEST")
    print("="*70)
    print(f"Model: {args.model}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer loaded (vocab: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return 1
    
    # Run tests
    results = {}
    
    if args.dataset == 'all':
        test_list = datasets_to_test.items()
    elif args.dataset in datasets_to_test:
        test_list = [(args.dataset, datasets_to_test[args.dataset])]
    else:
        print(f"\n✗ Unknown dataset: {args.dataset}")
        print(f"Available: {', '.join(datasets_to_test.keys())}")
        return 1
    
    for name, (cls, params) in test_list:
        results[name] = test_dataset(name, cls, params, tokenizer)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:20s} {status}")
    
    print(f"\nResult: {passed}/{total} datasets passed")
    print("="*70 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
