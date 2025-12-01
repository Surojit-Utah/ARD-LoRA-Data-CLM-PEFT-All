#!/usr/bin/env python3
"""
Test tokenizer differences between GPT-2 and LLaMA for ARC-Easy masking.
This shows the critical differences and cache implications.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

def compare_tokenizers():
    """Compare GPT-2 vs LLaMA tokenizer for answer choices."""
    
    print("TOKENIZER COMPARISON: GPT-2 vs LLaMA-2")
    print("="*60)
    
    # Load tokenizers
    try:
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    except Exception as e:
        print(f"ERROR: Failed to load GPT-2 tokenizer: {e}")
        return
    
    try:
        # Try to load LLaMA-2 tokenizer (or use available alternative)
        llama_tokenizer = None
        llama_attempts = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf", 
            "huggyllama/llama-7b",
            "microsoft/DialoGPT-medium"  # Fallback
        ]
        
        for model_name in llama_attempts:
            try:
                llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
                if llama_tokenizer.pad_token is None:
                    llama_tokenizer.pad_token = llama_tokenizer.eos_token
                print(f"[SUCCESS] Loaded tokenizer: {model_name}")
                break
            except Exception as e:
                print(f"[FAILED] {model_name}: {str(e)[:60]}...")
                continue
        
        if llama_tokenizer is None:
            print("[ERROR] Could not load any LLaMA tokenizer")
            return
            
    except Exception as e:
        print(f"ERROR: Failed to load LLaMA tokenizer: {e}")
        return
    
    # Test text - sample ARC-Easy question
    test_text = "Which dog trait is a learned behavior? A) blinking its eyes B) scratching an itch C) panting to cool off D) jumping to catch a ball The answer is D"
    
    print(f"\nTest text: {test_text}")
    print("\n" + "-"*60)
    
    # Tokenize with both
    gpt2_tokens = gpt2_tokenizer.encode(test_text, add_special_tokens=False)
    llama_tokens = llama_tokenizer.encode(test_text, add_special_tokens=False)
    
    print(f"\nGPT-2 Tokenization:")
    print(f"  Vocab size: {gpt2_tokenizer.vocab_size}")
    print(f"  Tokens: {len(gpt2_tokens)}")
    print(f"  Token IDs: {gpt2_tokens}")
    
    print(f"\nLLaMA Tokenization:")
    print(f"  Vocab size: {llama_tokenizer.vocab_size}")
    print(f"  Tokens: {len(llama_tokens)}")
    print(f"  Token IDs: {llama_tokens}")
    
    # Critical: Check answer choice patterns
    print(f"\n" + "="*60)
    print("ANSWER CHOICE PATTERN ANALYSIS")
    print("="*60)
    
    choices = ['A', 'B', 'C', 'D']
    patterns_to_test = [
        lambda c: c,                    # 'A'
        lambda c: f" {c}",             # ' A'  
        lambda c: f"({c})",            # '(A)'
        lambda c: f" ({c})",           # ' (A)'
        lambda c: f"{c})",             # 'A)'
        lambda c: f" {c})",            # ' A)'
    ]
    
    for choice in choices:
        print(f"\nChoice '{choice}' patterns:")
        for i, pattern_func in enumerate(patterns_to_test):
            pattern_text = pattern_func(choice)
            
            gpt2_pattern = gpt2_tokenizer.encode(pattern_text, add_special_tokens=False)
            llama_pattern = llama_tokenizer.encode(pattern_text, add_special_tokens=False)
            
            print(f"  Pattern '{pattern_text}':")
            print(f"    GPT-2:  {gpt2_pattern}")
            print(f"    LLaMA:  {llama_pattern}")
            
            if gpt2_pattern != llama_pattern:
                print(f"    ⚠️  DIFFERENT TOKENIZATION!")
    
    # Show final answer detection
    print(f"\n" + "="*60)
    print("FINAL ANSWER DETECTION")
    print("="*60)
    
    answer_phrases = ["The answer is D", " answer is D", "answer is D", " D"]
    
    for phrase in answer_phrases:
        gpt2_phrase = gpt2_tokenizer.encode(phrase, add_special_tokens=False)
        llama_phrase = llama_tokenizer.encode(phrase, add_special_tokens=False)
        
        print(f"\nPhrase '{phrase}':")
        print(f"  GPT-2:  {gpt2_phrase}")
        print(f"  LLaMA:  {llama_phrase}")
        
        if gpt2_phrase != llama_phrase:
            print(f"  ⚠️  DIFFERENT TOKENIZATION!")
    
    print(f"\n" + "="*60)
    print("IMPLICATIONS & RECOMMENDATIONS")
    print("="*60)
    print("1. CACHE SPECIFICITY:")
    print("   - Cache IS tokenizer-specific (different token IDs)")
    print("   - Need separate cache for each tokenizer")
    print("   - Pattern detection will be different")
    print()
    print("2. MASKING IMPLICATIONS:")
    print("   - Answer patterns will have different token IDs")
    print("   - Need to regenerate cache when switching tokenizers") 
    print("   - Backward search will find different positions")
    print()
    print("3. TRAINING IMPACT:")
    print("   - Labels/masks will be different between tokenizers")
    print("   - Model training will supervise different token positions")
    print("   - Results NOT directly comparable between tokenizers")

if __name__ == "__main__":
    compare_tokenizers()