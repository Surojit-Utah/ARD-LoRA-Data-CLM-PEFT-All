"""
Dataset debugging and filtering utilities for ARD-LoRA training.

This module contains utilities for analyzing and fixing dataset filtering issues
that can occur during training with DataCollatorForLanguageModeling.
"""

import random
import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


def debug_dataset_filtering(dataset, tokenizer, max_len=2048, sample_size=1000):
    """
    Debug dataset filtering to identify why samples are being lost.
    
    Args:
        dataset: The dataset to analyze
        tokenizer: Tokenizer used for processing
        max_len: Maximum sequence length
        sample_size: Number of samples to analyze
    
    Returns:
        Dictionary with filtering statistics
    """
    print("\n🔍 DEBUGGING DATASET FILTERING")
    print("=" * 50)
    
    total_samples = len(dataset)
    sample_size = min(sample_size, total_samples)
    
    # Analysis counters
    stats = {
        'total_samples': total_samples,
        'analyzed_samples': sample_size,
        'valid_samples': 0,
        'too_long_samples': 0,
        'empty_samples': 0,
        'tokenization_failed': 0,
        'other_issues': 0,
        'sequence_lengths': []
    }
    
    print(f"Analyzing {sample_size} samples from {total_samples} total samples...")
    
    # Sample random indices to analyze
    random.seed(42)
    sample_indices = random.sample(range(total_samples), sample_size)
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            
            # Check if sample has required fields
            if not sample or 'input_ids' not in sample:
                stats['empty_samples'] += 1
                continue
            
            # Get sequence length
            input_ids = sample['input_ids']
            if isinstance(input_ids, (list, tuple)):
                seq_len = len(input_ids)
            elif hasattr(input_ids, 'shape'):
                seq_len = input_ids.shape[-1] if len(input_ids.shape) > 0 else 0
            else:
                seq_len = 0
            
            stats['sequence_lengths'].append(seq_len)
            
            # Check sequence length
            if seq_len > max_len:
                stats['too_long_samples'] += 1
                continue
            elif seq_len == 0:
                stats['empty_samples'] += 1
                continue
            
            # Check labels if available
            if 'labels' in sample:
                labels = sample['labels']
                if isinstance(labels, (list, tuple)):
                    # Check if all labels are -100 (masked)
                    if hasattr(labels, '__iter__') and all(l == -100 for l in labels):
                        stats['empty_samples'] += 1
                        continue
            
            stats['valid_samples'] += 1
            
        except Exception as e:
            stats['tokenization_failed'] += 1
            if sample_size <= 10:  # Only print errors for small samples
                print(f"   Error processing sample {idx}: {e}")
    
    # Calculate statistics
    if stats['sequence_lengths']:
        seq_lengths = np.array(stats['sequence_lengths'])
        stats['avg_seq_length'] = np.mean(seq_lengths)
        stats['median_seq_length'] = np.median(seq_lengths)
        stats['max_seq_length'] = np.max(seq_lengths)
        stats['min_seq_length'] = np.min(seq_lengths)
        stats['seq_lengths_over_max'] = np.sum(seq_lengths > max_len)
    
    # Print results
    print(f"\n📊 FILTERING ANALYSIS RESULTS:")
    print(f"   Total samples: {stats['total_samples']:,}")
    print(f"   Analyzed samples: {stats['analyzed_samples']:,}")
    print(f"   Valid samples: {stats['valid_samples']:,} ({stats['valid_samples']/stats['analyzed_samples']*100:.1f}%)")
    print(f"   Too long (>{max_len}): {stats['too_long_samples']:,} ({stats['too_long_samples']/stats['analyzed_samples']*100:.1f}%)")
    print(f"   Empty/invalid: {stats['empty_samples']:,} ({stats['empty_samples']/stats['analyzed_samples']*100:.1f}%)")
    print(f"   Tokenization failed: {stats['tokenization_failed']:,} ({stats['tokenization_failed']/stats['analyzed_samples']*100:.1f}%)")
    
    if stats['sequence_lengths']:
        print(f"\n📏 SEQUENCE LENGTH STATISTICS:")
        print(f"   Average length: {stats['avg_seq_length']:.1f}")
        print(f"   Median length: {stats['median_seq_length']:.1f}")
        print(f"   Min length: {stats['min_seq_length']}")
        print(f"   Max length: {stats['max_seq_length']}")
        print(f"   Sequences > {max_len}: {stats['seq_lengths_over_max']:,}")
    
    return stats


def analyze_data_collator_filtering(dataset, tokenizer, batch_size=4, max_len=2048, num_batches=10):
    """
    Analyze how DataCollatorForLanguageModeling filters data.
    
    Args:
        dataset: Dataset to test
        tokenizer: Tokenizer to use
        batch_size: Batch size for testing
        max_len: Maximum sequence length
        num_batches: Number of batches to test
    
    Returns:
        Dictionary with collator filtering statistics
    """
    print("\n🔍 ANALYZING DATA COLLATOR FILTERING")
    print("=" * 50)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    stats = {
        'total_batches_attempted': 0,
        'successful_batches': 0,
        'failed_batches': 0,
        'total_samples_attempted': 0,
        'total_samples_in_successful_batches': 0,
        'errors': []
    }
    
    print(f"Testing {num_batches} batches with batch_size={batch_size}...")
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            stats['total_batches_attempted'] += 1
            stats['total_samples_attempted'] += batch_size
            
            try:
                # Check batch contents
                if 'input_ids' in batch:
                    input_ids = batch['input_ids']
                    batch_actual_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
                    stats['total_samples_in_successful_batches'] += batch_actual_size
                    stats['successful_batches'] += 1
                    
                    if batch_idx == 0:  # Print details for first batch
                        print(f"\n   First batch details:")
                        print(f"     Requested batch size: {batch_size}")
                        print(f"     Actual batch size: {batch_actual_size}")
                        print(f"     Input shape: {input_ids.shape if hasattr(input_ids, 'shape') else 'N/A'}")
                        if 'labels' in batch:
                            labels = batch['labels']
                            print(f"     Labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")
                else:
                    stats['failed_batches'] += 1
                    stats['errors'].append(f"Batch {batch_idx}: No input_ids in batch")
                    
            except Exception as e:
                stats['failed_batches'] += 1
                stats['errors'].append(f"Batch {batch_idx}: {str(e)}")
                
    except Exception as e:
        print(f"   DataLoader error: {e}")
        stats['errors'].append(f"DataLoader error: {str(e)}")
    
    # Calculate filtering rate
    if stats['total_samples_attempted'] > 0:
        filtering_rate = (stats['total_samples_attempted'] - stats['total_samples_in_successful_batches']) / stats['total_samples_attempted'] * 100
    else:
        filtering_rate = 0
    
    print(f"\n📊 DATA COLLATOR ANALYSIS RESULTS:")
    print(f"   Attempted batches: {stats['total_batches_attempted']}")
    print(f"   Successful batches: {stats['successful_batches']}")
    print(f"   Failed batches: {stats['failed_batches']}")
    print(f"   Total samples attempted: {stats['total_samples_attempted']}")
    print(f"   Samples in successful batches: {stats['total_samples_in_successful_batches']}")
    print(f"   Data collator filtering rate: {filtering_rate:.1f}%")
    
    if stats['errors']:
        print(f"\n⚠️  ERRORS ENCOUNTERED:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"     {error}")
        if len(stats['errors']) > 5:
            print(f"     ... and {len(stats['errors']) - 5} more errors")
    
    return stats


def get_effective_dataset_size(trainer):
    """
    Calculate the effective dataset size used by the trainer.
    
    Args:
        trainer: The ARDCLMTrainer instance
    
    Returns:
        Dictionary with dataset size information
    """
    print("\n📊 CALCULATING EFFECTIVE DATASET SIZE")
    print("=" * 50)
    
    # Get trainer's dataloader
    train_dataloader = trainer.get_train_dataloader()
    
    stats = {
        'reported_dataset_size': len(trainer.train_dataset) if trainer.train_dataset else 0,
        'dataloader_batch_count': len(train_dataloader) if train_dataloader else 0,
        'per_device_batch_size': trainer.args.per_device_train_batch_size,
        'gradient_accumulation_steps': trainer.args.gradient_accumulation_steps,
        'world_size': trainer.args.world_size if hasattr(trainer.args, 'world_size') else 1,
        'effective_batch_size': None,
        'effective_dataset_size': None,
        'filtering_percentage': None
    }
    
    # Calculate effective batch size
    stats['effective_batch_size'] = (
        stats['per_device_batch_size'] * 
        stats['gradient_accumulation_steps'] * 
        stats['world_size']
    )
    
    # Calculate effective dataset size
    if stats['dataloader_batch_count'] > 0:
        stats['effective_dataset_size'] = (
            stats['dataloader_batch_count'] * stats['effective_batch_size']
        )
        
        # Calculate filtering percentage
        if stats['reported_dataset_size'] > 0:
            stats['filtering_percentage'] = (
                (stats['reported_dataset_size'] - stats['effective_dataset_size']) / 
                stats['reported_dataset_size'] * 100
            )
    
    print(f"   Reported dataset size: {stats['reported_dataset_size']:,}")
    print(f"   DataLoader batch count: {stats['dataloader_batch_count']:,}")
    print(f"   Per-device batch size: {stats['per_device_batch_size']}")
    print(f"   Gradient accumulation steps: {stats['gradient_accumulation_steps']}")
    print(f"   World size: {stats['world_size']}")
    print(f"   Effective batch size: {stats['effective_batch_size']}")
    
    if stats['effective_dataset_size'] is not None:
        print(f"   Effective dataset size: {stats['effective_dataset_size']:,}")
        print(f"   Samples lost to filtering: {stats['reported_dataset_size'] - stats['effective_dataset_size']:,}")
        if stats['filtering_percentage'] is not None:
            print(f"   Filtering percentage: {stats['filtering_percentage']:.1f}%")
    
    return stats


def apply_dataset_filtering_fixes(dataset, tokenizer, max_len=2048, verbose=True):
    """
    Apply common fixes for dataset filtering issues.
    
    Args:
        dataset: Dataset to fix
        tokenizer: Tokenizer to use
        max_len: Maximum sequence length
        verbose: Whether to print progress
    
    Returns:
        Fixed dataset
    """
    if verbose:
        print("\n🔧 APPLYING DATASET FILTERING FIXES")
        print("=" * 50)
        print(f"Original dataset size: {len(dataset):,}")
    
    # Fix 1: Remove samples that are too long
    def filter_length(example):
        if 'input_ids' in example:
            input_ids = example['input_ids']
            if isinstance(input_ids, (list, tuple)):
                return len(input_ids) <= max_len
            elif hasattr(input_ids, 'shape'):
                return input_ids.shape[-1] <= max_len
        return True
    
    # Fix 2: Remove empty samples
    def filter_empty(example):
        if 'input_ids' not in example:
            return False
        input_ids = example['input_ids']
        if isinstance(input_ids, (list, tuple)):
            return len(input_ids) > 0
        elif hasattr(input_ids, 'shape'):
            return input_ids.shape[-1] > 0
        return True
    
    # Fix 3: Remove samples with all masked labels
    def filter_all_masked(example):
        if 'labels' in example:
            labels = example['labels']
            if isinstance(labels, (list, tuple)):
                return not all(l == -100 for l in labels)
            elif hasattr(labels, '__iter__'):
                try:
                    return not all(l == -100 for l in labels)
                except:
                    return True
        return True
    
    # Apply filters if dataset supports filtering
    if hasattr(dataset, 'filter'):
        if verbose:
            print("   Applying length filter...")
        dataset = dataset.filter(filter_length)
        if verbose:
            print(f"   After length filter: {len(dataset):,}")
        
        if verbose:
            print("   Applying empty sample filter...")
        dataset = dataset.filter(filter_empty)
        if verbose:
            print(f"   After empty filter: {len(dataset):,}")
        
        if verbose:
            print("   Applying masked labels filter...")
        dataset = dataset.filter(filter_all_masked)
        if verbose:
            print(f"   After masked labels filter: {len(dataset):,}")
    else:
        if verbose:
            print("   Dataset doesn't support filtering - manual filtering needed")
    
    if verbose:
        print(f"\n✅ Dataset filtering fixes applied")
        print(f"   Final dataset size: {len(dataset):,}")
    
    return dataset


def run_complete_dataset_analysis(train_dataset, eval_dataset, tokenizer, cfg):
    """
    Run a complete dataset analysis including all debugging functions.
    
    Args:
        train_dataset: Training dataset to analyze
        eval_dataset: Evaluation dataset to analyze (optional)
        tokenizer: Tokenizer used for processing
        cfg: Configuration dictionary
    
    Returns:
        Dictionary with all analysis results
    """
    results = {}
    
    print("\n🔍 COMPLETE DATASET ANALYSIS")
    print("=" * 60)
    
    # Debug training dataset
    if train_dataset is not None:
        print("\n📊 TRAINING DATASET ANALYSIS:")
        results['train_debug'] = debug_dataset_filtering(
            train_dataset, 
            tokenizer, 
            max_len=cfg["max_len"],
            sample_size=min(1000, len(train_dataset))
        )
        
        # Test data collator
        results['train_collator'] = analyze_data_collator_filtering(
            train_dataset,
            tokenizer,
            batch_size=cfg["batch_size"],
            max_len=cfg["max_len"],
            num_batches=10
        )
    
    # Debug validation dataset
    if eval_dataset is not None:
        print("\n📊 VALIDATION DATASET ANALYSIS:")
        results['eval_debug'] = debug_dataset_filtering(
            eval_dataset, 
            tokenizer, 
            max_len=cfg["max_len"],
            sample_size=min(250, len(eval_dataset))
        )
    
    return results