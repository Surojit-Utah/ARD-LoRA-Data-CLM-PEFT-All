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


def deep_dive_single_example(dataset_key='winogrande_s', example_idx=0, tokenizer=None):
    """
    Deep dive into processing a single example to show complete transformation pipeline.
    
    Args:
        dataset_key: Dataset to use (default: winogrande_s)
        example_idx: Index of example to analyze (default: 0)
        tokenizer: Tokenizer to use
    """
    print_section_header(f"DEEP DIVE: Single Example Processing")
    
    config = DATASET_CONFIGS[dataset_key]
    
    print(f"Dataset: {config['name']}")
    print(f"Example Index: {example_idx}")
    print()
    
    # Initialize dataset
    print("STEP 1: Initialize Dataset")
    print("-" * 70)
    dataset_obj = config['class'](
        tokenizer=tokenizer,
        add_space=True,
        **config['init_params']
    )
    print(f"✓ Dataset class: {config['class'].__name__}")
    print(f"✓ Number of labels: {dataset_obj.n_labels}")
    print(f"✓ Preamble template loaded")
    print()
    
    # Show label to token mappings
    print("STEP 2: Label-to-Token Mappings (created during __init__)")
    print("-" * 70)
    print("label2target dictionary:")
    for label_idx, token_id in dataset_obj.label2target.items():
        token_str = tokenizer.convert_ids_to_tokens(token_id.item())
        token_text = tokenizer.decode(token_id)
        print(f"  {label_idx} -> token_id={token_id.item()}, token={repr(token_str)}, decoded={repr(token_text)}")
    print()
    
    # Get raw example
    print("STEP 3: Get Raw Example from HuggingFace Dataset")
    print("-" * 70)
    raw_example = dataset_obj.dset['train'][example_idx]
    print("Raw dictionary fields:")
    for key, value in raw_example.items():
        print(f"  {key}: {repr(value)}")
    print()
    
    # Format prompt
    print("STEP 4: Format Prompt (_format_prompts)")
    print("-" * 70)
    print("Building choices string:")
    choices_str = f"A) {raw_example['option1']}\nB) {raw_example['option2']}"
    print(f"  choices = 'A) {raw_example['option1']}\\nB) {raw_example['option2']}'")
    print()
    print("Filling preamble template:")
    print(f"  preamble.format(question={repr(raw_example['sentence'])}, choices=...)")
    print()
    
    formatted_prompts = dataset_obj._format_prompts([raw_example])
    prompt_string = formatted_prompts[0]
    
    print("Resulting prompt string:")
    print("  " + "─" * 66)
    for line in prompt_string.split('\n'):
        print(f"  {line}")
    print("  " + "─" * 66)
    print()
    
    # Tokenize
    print("STEP 5: Tokenize Prompt (_tokenize_prompts)")
    print("-" * 70)
    tokenized = dataset_obj._tokenize_prompts(formatted_prompts)
    input_ids = tokenized['input_ids'][0]
    attention_mask = tokenized['attention_mask'][0]
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs tensor: {input_ids}")
    print()
    print(f"Attention mask: {attention_mask}")
    print()
    print("Token-by-token breakdown (first 20 tokens):")
    for i, token_id in enumerate(input_ids[:20].tolist()):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        print(f"  Position {i:2d}: id={token_id:5d}, token={repr(token_str):15s}")
    print(f"  ... ({len(input_ids)} tokens total)")
    print()
    print("Last 5 tokens (should end with 'Answer:'):")
    for i, token_id in enumerate(input_ids[-5:].tolist(), start=len(input_ids)-5):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        print(f"  Position {i:2d}: id={token_id:5d}, token={repr(token_str):15s}")
    print()
    
    # Extract class
    print("STEP 6: Extract Class Index (from answer field)")
    print("-" * 70)
    answer_raw = raw_example['answer']
    print(f"Raw answer field: {repr(answer_raw)} (type: {type(answer_raw).__name__})")
    print(f"Convert to int: int('{answer_raw}') = {int(answer_raw)}")
    print(f"Subtract 1 for zero-indexing: {int(answer_raw)} - 1 = {int(answer_raw) - 1}")
    
    class_idx = int(answer_raw) - 1
    print(f"Final class index: {class_idx}")
    print()
    
    if class_idx == 0:
        print(f"  → This means option A ('{raw_example['option1']}') is correct")
    else:
        print(f"  → This means option B ('{raw_example['option2']}') is correct")
    print()
    
    # Map to target
    print("STEP 7: Map Class Index to Target Token ID")
    print("-" * 70)
    print(f"Looking up label2target[{class_idx}]:")
    target_tensor = dataset_obj.label2target[class_idx]
    target_id = target_tensor.item()
    print(f"  label2target[{class_idx}] = tensor([[{target_id}]])")
    print()
    print(f"Target token ID: {target_id}")
    print(f"Target token (raw): {repr(tokenizer.convert_ids_to_tokens(target_id))}")
    print(f"Target token (decoded): {repr(tokenizer.decode([target_id]))}")
    print()
    
    # Complete collate_fn
    print("STEP 8: Complete clm_collate_fn Processing")
    print("-" * 70)
    batch = [raw_example]
    prompts, classes, targets = dataset_obj.clm_collate_fn(batch)
    
    print("Return values from clm_collate_fn:")
    print()
    print("1. prompts (dict):")
    print(f"   input_ids shape: {prompts['input_ids'].shape}")
    print(f"   attention_mask shape: {prompts['attention_mask'].shape}")
    print()
    print("2. classes (tensor):")
    print(f"   {classes}")
    print(f"   Value: {classes[0].item()}")
    print()
    print("3. targets (tensor):")
    print(f"   {targets}")
    print(f"   Value: {targets[0].item()}")
    print()
    
    # Model prediction scenario
    print("STEP 9: What the Model Does")
    print("-" * 70)
    print("The model receives:")
    print(f"  - input_ids ending with 'Answer:' (last token ID: {input_ids[-1].item()})")
    print(f"  - attention_mask (all 1s for this example)")
    print()
    print("The model should predict:")
    print(f"  - Next token after 'Answer:' should be token {target_id}")
    print(f"  - This decodes to {repr(tokenizer.decode([target_id]))}")
    print(f"  - Which represents class {class_idx} ({'A' if class_idx == 0 else 'B'})")
    print()
    print("Loss computation:")
    print(f"  - Model outputs logits for all {tokenizer.vocab_size} vocabulary tokens")
    print(f"  - We only care about logits for answer tokens: {list(dataset_obj.label2target.values())}")
    print(f"  - Cross-entropy loss compares predicted distribution vs ground truth class {class_idx}")
    print()
    
    print_separator("=")
    print()
    
    
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

    parser.add_argument(
        '--deep-dive',
        action='store_true',
        help='Run deep dive analysis on a single example'
    )
    parser.add_argument(
        '--example-idx',
        type=int,
        default=0,
        help='Index of example for deep dive (default: 0)'
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
    if not args.deep_dive:
        print(f"Datasets: {', '.join(datasets_to_sample)}")
        print(f"Samples per split: {args.n_samples}")
    else:
        print(f"Mode: Deep dive on winogrande_s example {args.example_idx}")
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
    
    # Deep dive if requested
    if args.deep_dive:
        deep_dive_single_example(
            dataset_key='winogrande_s',
            example_idx=args.example_idx,
            tokenizer=tokenizer
        )
        return 0
    
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
