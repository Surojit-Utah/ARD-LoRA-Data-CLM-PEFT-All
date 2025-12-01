#!/usr/bin/env python3
"""
Simple test script showing only first 5 ARC-Easy samples with clean output.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.bayesian_peft_cached import load_bayesian_peft_with_caching

def test_first_5_samples():
    """Test only the first 5 ARC-Easy samples with minimal output."""
    
    print("ARC-EASY MASKING TEST - FIRST 5 SAMPLES")
    print("="*50)
    
    # Configuration
    config = {"max_length": 512}
    cache_root = "G:/My Drive/ARD_LoRA_Data_Cache"
    
    # Load tokenizer quietly
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        return
    
    # Load datasets quietly
    try:
        # Capture dataset output in background
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            train_ds, val_ds, tokenizer = load_bayesian_peft_with_caching(
                dataset_name="arc_easy",
                tokenizer_name="gpt2",
                config=config,
                cache_root=cache_root
            )
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return
    
    # Analyze exactly 5 samples
    for i in range(5):
        sample = train_ds[i]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # Count unmasked tokens (not -100)
        unmasked_indices = [j for j, label in enumerate(labels) if label != -100]
        unmasked_count = len(unmasked_indices)
        
        # Count actual content (non-padding tokens)
        content_length = sum(1 for token_id in input_ids if token_id != tokenizer.pad_token_id)
        actual_content = input_ids[:content_length]
        
        # Decode full text (complete question with all options)
        full_text = tokenizer.decode(actual_content, skip_special_tokens=True)
        
        print(f"\nSample {i+1}:")
        print(f"Complete text: {full_text}")
        print(f"Total content tokens: {content_length}")
        print(f"Unmasked tokens: {unmasked_count}")
        
        # Find all A/B/C/D option positions in the sequence
        choice_positions = {}
        for choice in ['A', 'B', 'C', 'D']:
            positions = []
            # Check for different patterns: 'A', ' A', '(A)', ' (A)'
            patterns = [
                tokenizer.encode(choice, add_special_tokens=False),
                tokenizer.encode(f" {choice}", add_special_tokens=False),
                tokenizer.encode(f"({choice})", add_special_tokens=False),
                tokenizer.encode(f" ({choice})", add_special_tokens=False),
            ]
            
            for pattern in patterns:
                if pattern:  # Only check non-empty patterns
                    for start_pos in range(len(actual_content) - len(pattern) + 1):
                        if actual_content[start_pos:start_pos + len(pattern)] == pattern:
                            pattern_text = tokenizer.decode(pattern, skip_special_tokens=True)
                            positions.append((start_pos, start_pos + len(pattern) - 1, pattern_text))
            
            if positions:
                choice_positions[choice] = positions
        
        # Show all option positions
        print("Option positions found:")
        for choice, positions in choice_positions.items():
            for start, end, text in positions:
                print(f"  {choice}: tokens [{start}:{end+1}] = '{text}'")
        
        # Analyze unmasked tokens
        if unmasked_indices:
            unmasked_tokens = [input_ids[idx] for idx in unmasked_indices]
            unmasked_text = tokenizer.decode(unmasked_tokens, skip_special_tokens=True).strip()
            print(f"Unmasked answer: '{unmasked_text}' at positions {unmasked_indices}")
            
            # Validate: check which option this corresponds to
            correct_choice = None
            for choice, positions in choice_positions.items():
                for start, end, text in positions:
                    # Check if unmasked position overlaps with this choice
                    if any(start <= pos <= end for pos in unmasked_indices):
                        correct_choice = choice
                        # Check if it's the LAST occurrence (final answer)
                        is_final = all(pos >= max(p[1] for p in choice_positions.get(choice, [])) - 1 for pos in unmasked_indices)
                        print(f"  → Matches choice '{choice}' at position [{start}:{end+1}] ({'FINAL answer' if is_final else 'choice option'})")
                        break
            
            # Validation status
            if unmasked_count == 1 and correct_choice and unmasked_text in ['A', 'B', 'C', 'D']:
                print("Status: ✓ PERFECT - exactly 1 answer token, correctly identified")
            else:
                print(f"Status: ❌ ISSUE - expected 1 A/B/C/D token, got {unmasked_count}")
        else:
            print("Status: ❌ ERROR - no unmasked tokens")
        
        # Show context around unmasked token
        if unmasked_indices:
            pos = unmasked_indices[0]
            context_start = max(0, pos - 5)
            context_end = min(content_length, pos + 6)
            context_tokens = actual_content[context_start:context_end]
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
            print(f"Context around answer: '...{context_text}...'")
        
        print("-" * 80)
    
    print("\n" + "="*50)
    print("Summary: ARC-Easy answer masking validation complete")

if __name__ == "__main__":
    test_first_5_samples()