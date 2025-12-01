"""
Dataset Sampling Test Script
============================

This script samples and displays examples from multiple datasets to verify:
1. Dataset loading works correctly
2. Formatting and prompts are appropriate
3. Label extraction is correct
4. Data structure matches expected format

Supports the following datasets:
- WG-S: WinoGrande-S (AllenAI)
- WG-M: WinoGrande-M (AllenAI)
- ARC-C: ARC-Challenge (AI2)
- ARC-E: ARC-Easy (AI2)
- OBQA: OpenBookQA (AI2)
- RTE: Recognizing Textual Entailment (GLUE)
- MRPC: Microsoft Research Paraphrase Corpus (GLUE)
- CoLA: Corpus of Linguistic Acceptability (GLUE)
- BoolQ: Boolean Questions (SuperGLUE)
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import dataset modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from dataset.utils import dsets


# Dataset configurations
DATASET_CONFIGS = {
    'winogrande_s': {
        'class': dsets.WinograndeDataset,
        'name': 'WinoGrande-S',
        'num_labels': 2,
        'label_type': 'MCQ',  # A/B
        'init_params': {'name': 'winogrande_s'},
        'splits': ['train', 'validation']
    },
    'winogrande_m': {
        'class': dsets.WinograndeDataset,
        'name': 'WinoGrande-M',
        'num_labels': 2,
        'label_type': 'MCQ',  # A/B
        'init_params': {'name': 'winogrande_m'},
        'splits': ['train', 'validation']
    },
    'arc_easy': {
        'class': dsets.ARCDataset,
        'name': 'ARC-Easy',
        'num_labels': 5,
        'label_type': 'MCQ',  # A/B/C/D/E
        'init_params': {'name': 'ARC-Easy'},
        'splits': ['train', 'validation', 'test']
    },
    'arc_challenge': {
        'class': dsets.ARCDataset,
        'name': 'ARC-Challenge',
        'num_labels': 5,
        'label_type': 'MCQ',  # A/B/C/D/E
        'init_params': {'name': 'ARC-Challenge'},
        'splits': ['train', 'validation', 'test']
    },
    'obqa': {
        'class': dsets.OBQADataset,
        'name': 'OpenBookQA',
        'num_labels': 4,
        'label_type': 'MCQ',  # A/B/C/D
        'init_params': {},
        'splits': ['train', 'validation', 'test']
    },
    'rte': {
        'class': dsets.RTEDataset,
        'name': 'RTE (Recognizing Textual Entailment)',
        'num_labels': 2,
        'label_type': 'Binary',  # 0/1
        'init_params': {},
        'splits': ['train', 'validation', 'test']
    },
    'mrpc': {
        'class': dsets.MRPCDataset,
        'name': 'MRPC (Paraphrase Corpus)',
        'num_labels': 2,
        'label_type': 'Binary',  # 0/1
        'init_params': {},
        'splits': ['train', 'validation', 'test']
    },
    'cola': {
        'class': dsets.CoLADataset,
        'name': 'CoLA (Linguistic Acceptability)',
        'num_labels': 2,
        'label_type': 'Binary',  # 0/1
        'init_params': {},
        'splits': ['train', 'validation', 'test']
    },
    'boolq': {
        'class': dsets.BoolQDataset,
        'name': 'BoolQ (Boolean Questions)',
        'num_labels': 2,
        'label_type': 'Boolean',  # True/False -> 0/1
        'init_params': {},
        'splits': ['train', 'validation']
    },
}


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section_header(title):
    """Print a formatted section header."""
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def sample_dataset(dataset_key, tokenizer, n_samples=3, seed=42):
    """
    Sample examples from a dataset and display formatted output.
    
    Args:
        dataset_key: Key identifying the dataset in DATASET_CONFIGS
        tokenizer: Tokenizer to use for the dataset
        n_samples: Number of samples to display
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    config = DATASET_CONFIGS[dataset_key]
    
    print_section_header(f"DATASET: {config['name']}")
    
    print(f"Dataset Key:   {dataset_key}")
    print(f"Label Type:    {config['label_type']}")
    print(f"Num Labels:    {config['num_labels']}")
    print(f"Available Splits: {', '.join(config['splits'])}")
    print()
    
    try:
        # Initialize dataset
        print(f"Loading dataset class: {config['class'].__name__}...")
        dataset_obj = config['class'](
            tokenizer=tokenizer,
            add_space=True,
            **config['init_params']
        )
        
        print(f"✓ Dataset loaded successfully")
        print()
        
        # Display target token mappings
        print("Target Token Mappings (label_index -> token_id):")
        print("-" * 50)
        for label_idx, token_id in dataset_obj.label2target.items():
            token_str = tokenizer.convert_ids_to_tokens(token_id.item())
            token_text = tokenizer.decode(token_id)
            print(f"  Label {label_idx}: token_id={token_id.item():5d}, token={repr(token_str):10s}, text={repr(token_text)}")
        print()
        
        # Sample from each available split
        for split in config['splits']:
            if split not in dataset_obj.dset:
                print(f"⚠ Split '{split}' not available in dataset")
                continue
            
            split_data = dataset_obj.dset[split]
            split_size = len(split_data)
            
            print(f"--- Split: {split.upper()} (Total samples: {split_size}) ---")
            print()
            
            if split_size == 0:
                print(f"⚠ Split '{split}' is empty")
                print()
                continue
            
            # Sample random indices
            sample_size = min(n_samples, split_size)
            sample_indices = random.sample(range(split_size), sample_size)
            
            # Create a small batch for collate_fn
            batch = [split_data[idx] for idx in sample_indices]
            
            # Process through collate_fn to see what the model receives
            try:
                prompts, classes, targets = dataset_obj.clm_collate_fn(batch)
                
                # Display each example
                for i, (idx, example) in enumerate(zip(sample_indices, batch)):
                    print(f"Example {i+1}/{sample_size} (Index: {idx})")
                    print("-" * 70)
                    
                    # Display raw example fields
                    print("RAW DATA:")
                    display_raw_example(example, config['label_type'])
                    print()
                    
                    # Display processed prompt
                    print("PROCESSED FOR MODEL:")
                    prompt_text = tokenizer.decode(prompts['input_ids'][i], skip_special_tokens=True)
                    
                    # Truncate if too long
                    if len(prompt_text) > 500:
                        print(f"Prompt (truncated): {prompt_text[:500]}...")
                    else:
                        print(f"Prompt: {prompt_text}")
                    print()
                    
                    # Display label information
                    class_idx = classes[i].item() if torch.is_tensor(classes[i]) else classes[i]
                    target_token_id = targets[i].item() if torch.is_tensor(targets[i]) else targets[i]
                    target_token_str = tokenizer.convert_ids_to_tokens(target_token_id)
                    target_text = tokenizer.decode(target_token_id)
                    
                    print(f"Label (class index): {class_idx}")
                    print(f"Target token ID: {target_token_id}")
                    print(f"Target token: {repr(target_token_str)}")
                    print(f"Target text: {repr(target_text)}")
                    
                    # Show expected answer format
                    if config['label_type'] == 'MCQ':
                        if config['num_labels'] == 2:
                            expected = chr(ord('A') + class_idx)  # A or B
                        else:
                            expected = chr(ord('A') + class_idx)  # A/B/C/D/E
                        print(f"Expected answer: {expected}")
                    elif config['label_type'] in ['Binary', 'Boolean']:
                        print(f"Expected answer: {class_idx}")
                    
                    print()
                    print("=" * 70)
                    print()
            
            except Exception as e:
                print(f"✗ Error processing batch from split '{split}': {e}")
                import traceback
                traceback.print_exc()
            
            print()
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print()


def display_raw_example(example, label_type):
    """
    Display raw example data in a readable format.
    
    Args:
        example: Raw example dictionary from dataset
        label_type: Type of label (MCQ, Binary, Boolean)
    """
    # Common fields to display
    common_fields = ['question', 'sentence', 'sentence1', 'sentence2', 
                     'passage', 'hypothesis', 'premise', 'word']
    
    for field in common_fields:
        if field in example and example[field]:
            value = example[field]
            # Truncate if too long
            if isinstance(value, str) and len(value) > 200:
                print(f"  {field.capitalize()}: {value[:200]}...")
            else:
                print(f"  {field.capitalize()}: {value}")
    
    # Display choices for MCQ datasets
    if label_type == 'MCQ' and 'choices' in example:
        print(f"  Choices:")
        choices = example['choices']
        if isinstance(choices, dict):
            # Format: {'text': [...], 'label': [...]}
            if 'text' in choices and 'label' in choices:
                for label, text in zip(choices['label'], choices['text']):
                    print(f"    {label}) {text}")
        elif isinstance(choices, list):
            # Format: list of strings or dict
            for i, choice in enumerate(choices):
                if isinstance(choice, str):
                    print(f"    {chr(ord('A')+i)}) {choice}")
                elif isinstance(choice, dict):
                    print(f"    {chr(ord('A')+i)}) {choice}")
    
    # Display options for Winogrande
    if 'option1' in example and 'option2' in example:
        print(f"  Options:")
        print(f"    A) {example['option1']}")
        print(f"    B) {example['option2']}")
    
    # Display answer/label
    if 'answerKey' in example:
        print(f"  Answer Key: {example['answerKey']}")
    elif 'answer' in example:
        print(f"  Answer: {example['answer']}")
    elif 'label' in example:
        print(f"  Label: {example['label']}")


def main():
    """Main function to run dataset sampling tests."""
    parser = argparse.ArgumentParser(
        description="Sample and display examples from multiple datasets"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASET_CONFIGS.keys()) + ['all'],
        default=['all'],
        help='Datasets to sample from (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        help='Model/tokenizer to use (default: meta-llama/Llama-2-7b-hf)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=3,
        help='Number of samples per split (default: 3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save results (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to sample
    if 'all' in args.datasets:
        datasets_to_sample = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_sample = args.datasets
    
    # Print header
    print()
    print_separator("=")
    print("  DATASET SAMPLING TEST SCRIPT")
    print_separator("=")
    print()
    print(f"Model/Tokenizer: {args.model}")
    print(f"Datasets: {', '.join(datasets_to_sample)}")
    print(f"Samples per split: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  PAD token: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")
        print(f"  EOS token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
        print()
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return 1
    
    # Redirect output if specified
    original_stdout = sys.stdout
    if args.output:
        print(f"Saving output to: {args.output}")
        output_file = open(args.output, 'w', encoding='utf-8')
        sys.stdout = output_file
        
        # Re-print header to file
        print()
        print_separator("=")
        print("  DATASET SAMPLING TEST SCRIPT")
        print_separator("=")
        print()
        print(f"Model/Tokenizer: {args.model}")
        print(f"Datasets: {', '.join(datasets_to_sample)}")
        print(f"Samples per split: {args.n_samples}")
        print(f"Random seed: {args.seed}")
        print()
    
    # Sample each dataset
    for dataset_key in datasets_to_sample:
        try:
            sample_dataset(
                dataset_key=dataset_key,
                tokenizer=tokenizer,
                n_samples=args.n_samples,
                seed=args.seed
            )
        except KeyboardInterrupt:
            print("\nSampling interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error sampling {dataset_key}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Restore stdout and close file
    if args.output:
        sys.stdout = original_stdout
        output_file.close()
        print(f"✓ Results saved to: {args.output}")
    
    print()
    print_separator("=")
    print("  SAMPLING COMPLETE")
    print_separator("=")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
